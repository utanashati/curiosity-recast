import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_atari_env, create_doom_env
from model import ActorCritic, IntrinsicCuriosityModule

from itertools import chain  # ICM

import os
import signal


class Killer:
    kill_now = False

    def __init__(self):
        signal.signal(signal.SIGTERM, self.exit)

    def exit(self, signum, frame):
        self.kill_now = True


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train_curiosity(
    rank, args, shared_model, shared_curiosity,
    counter, lock, pids, optimizer
):
    pids.append(os.getpid())

    torch.manual_seed(args.seed + rank)

    if args.game == 'doom':
        env = create_doom_env(
            args.env_name, rank,
            num_skip=args.num_skip, num_stack=args.num_stack)
    elif args.game == 'atari':
        env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(
        # env.observation_space.shape[0],
        args.num_stack,
        env.action_space)
    curiosity = IntrinsicCuriosityModule(  # ICM
        # env.observation_space.shape[0],
        args.num_stack,
        env.action_space)

    if optimizer is None:
        # optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
        optimizer = optim.Adam(  # ICM
            chain(shared_model.parameters(), shared_curiosity.parameters()),
            lr=args.lr)

    model.train()
    curiosity.train()  # ICM

    model.load_state_dict(shared_model.state_dict())

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0

    killer = Killer()
    while not killer.kill_now:
        # Sync with the shared model
        curiosity.load_state_dict(shared_curiosity.state_dict())  # ICM
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        inv_loss = torch.tensor(0.0)        # ICM
        forw_loss = torch.tensor(0.0)       # ICM

        for step in range(args.num_steps):
            if done:
                episode_length = 0
                state = env.reset()
                state = torch.from_numpy(state)
            episode_length += 1

            value, logit, (hx, cx) = model(state.unsqueeze(0),
                                           hx, cx)
            prob = F.softmax(logit, dim=-1)

            action = prob.multinomial(num_samples=1).flatten().detach()

            state_old = state  # ICM

            state, external_reward, done, _ = env.step(action.numpy())
            state = torch.from_numpy(state)

            # external reward = 0 if ICM-only mode
            external_reward = external_reward * (1 - args.icm_only)

            # <---ICM---
            inv_out, forw_out, curiosity_reward = \
                curiosity(
                    state_old.unsqueeze(0), action,
                    state.unsqueeze(0))
            # In noreward-rl:
            # self.invloss = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits, aindex),
            #     name="invloss")
            # self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
            # self.forwardloss = self.forwardloss * 288.0 # lenFeatures=288. Factored out to make hyperparams not depend on it.
            current_inv_loss = F.nll_loss(F.log_softmax(inv_out), action)
            # prob_curiosity = F.softmax(inv_out, dim=-1)
            # log_prob_curiosity = F.log_softmax(logit.detach(), dim=-1)
            # current_inv_loss = -(log_prob_curiosity * prob_curiosity).sum()
            current_forw_loss = curiosity_reward
            inv_loss += current_inv_loss
            forw_loss += current_forw_loss
            # ---ICM--->

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                break

        # <---ICM---
        inv_loss = inv_loss / episode_length
        forw_loss = forw_loss * (32 * 3 * 3) * 0.5 / episode_length

        curiosity_loss = args.lambda_1 * (
            (1 - args.beta) * inv_loss + args.beta * forw_loss)
        # ---ICM--->

        optimizer.zero_grad()

        curiosity_loss.backward()  # ICM
        torch.nn.utils.clip_grad_norm_(curiosity.parameters(), args.max_grad_norm)

        ensure_shared_grads(curiosity, shared_curiosity)
        optimizer.step()

    env.close()
