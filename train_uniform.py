import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_picolmaze_env
from model import IntrinsicCuriosityModule2

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


def train_uniform(
    rank, args, shared_curiosity, counter,
    lock, pids, optimizer, train_inv_losses,
    train_forw_losses
):
    pids.append(os.getpid())

    torch.manual_seed(args.seed + rank)

    env = create_picolmaze_env(args.num_rooms, args.colors)
    env.seed(args.seed + rank)

    curiosity = IntrinsicCuriosityModule2(  # ICM
        args.num_stack,
        env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(  # ICM
            chain(shared_curiosity.parameters()),
            lr=args.lr)

    curiosity.train()  # ICM

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

        inv_loss = torch.tensor(0.0)   # ICM
        forw_loss = torch.tensor(0.0)  # ICM
        curiosity_reward = torch.tensor(0.0)  # ICM

        for step in range(args.num_steps):
            if done:
                episode_length = 0
                state = env.reset()
                state = torch.from_numpy(state)
            episode_length += 1

            logit = torch.ones(1, env.action_space.n)
            prob = F.softmax(logit, dim=-1)

            action = prob.multinomial(num_samples=1).flatten().detach()

            state_old = state  # ICM

            state, _, done, _ = env.step(action)
            state = torch.from_numpy(state)

            # <---ICM---
            inv_out, phi2, forw_out_mean, forw_out_std, l2_loss, \
                bayesian_loss, current_curiosity_reward = \
                curiosity(
                    state_old.unsqueeze(0), action,
                    state.unsqueeze(0))
            # In noreward-rl:
            # self.invloss = tf.reduce_mean(
            #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits, aindex),
            #     name="invloss")
            # self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
            # self.forwardloss = self.forwardloss * 288.0 # lenFeatures=288. Factored out to make hyperparams not depend on it.
            current_inv_loss = F.nll_loss(F.log_softmax(inv_out, dim=-1), action)

            if args.new_curiosity:
                current_forw_loss = bayesian_loss
            else:
                current_forw_loss = l2_loss
                current_curiosity_reward = l2_loss

            inv_loss += current_inv_loss
            forw_loss += current_forw_loss
            curiosity_reward += current_curiosity_reward
            # ---ICM--->

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                break

        # <---ICM---
        inv_loss = inv_loss / episode_length
        # lenFeatures=288. Factored out to make hyperparams not depend on it.
        # NB: To me, it seems that it only makes hyperparameters DEPEND on it.
        # forw_loss = forw_loss * (32 * 3 * 3) * 0.5 / episode_length
        forw_loss = forw_loss / episode_length
        curiosity_reward = curiosity_reward / episode_length

        curiosity_loss = (1 - args.beta) * inv_loss + args.beta * forw_loss
        # ---ICM--->

        optimizer.zero_grad()

        train_inv_losses[rank - 1] = float((inv_loss).detach().item())
        train_forw_losses[rank - 1] = float((forw_loss).detach().item())

        curiosity_loss.backward()  # ICM
        torch.nn.utils.clip_grad_norm_(curiosity.parameters(), args.max_grad_norm)
        ensure_shared_grads(curiosity, shared_curiosity)
        optimizer.step()

    env.close()
