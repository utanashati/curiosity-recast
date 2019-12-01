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


def train(
    rank, args, shared_model, shared_curiosity,
    counter, lock, pids, optimizer=None
):
    pids.append(os.getpid())

    torch.manual_seed(args.seed + rank)

    if args.game == 'doom':
        env = create_doom_env(args.env_name, rank)
    elif args.game == 'atari':
        env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    curiosity = IntrinsicCuriosityModule(  # ICM
        env.observation_space.shape[0],
        env.action_space)

    if optimizer is None:
        # optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
        optimizer = optim.Adam(  # ICM
            chain(shared_model.parameters(), shared_curiosity.parameters()),
            lr=args.lr)

    model.train()
    curiosity.train()  # ICM

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0

    killer = Killer()
    while not killer.kill_now:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        curiosity.load_state_dict(shared_curiosity.state_dict())  # ICM
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        inv_loss = 0  # ICM
        forw_loss = 0  # ICM

        for step in range(args.num_steps):
            episode_length += 1
            value, logit, (hx, cx) = model((state.unsqueeze(0),
                                            (hx, cx)))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state_old = state  # ICM

            state, external_reward, done, _ = env.step(action.numpy())
            state = torch.from_numpy(state)  # Moved for ICM

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
            prob_curiosity = F.softmax(inv_out, dim=-1)
            log_prob_curiosity = F.log_softmax(logit.detach(), dim=-1)

            inv_loss += -(log_prob_curiosity * prob_curiosity).sum(
                1, keepdim=True)
            forw_loss += curiosity_reward

            curiosity_reward = args.eta * curiosity_reward

            reward = max(min(external_reward, args.clip), -args.clip) + \
                max(min(curiosity_reward.detach(), args.clip), -args.clip)
            # ---ICM--->

            done = done or episode_length >= args.max_episode_length

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()
                state = torch.from_numpy(state)  # ICM

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        # <---ICM---
        inv_loss = inv_loss / args.num_steps
        forw_loss = forw_loss * (32 * 3 * 3) * 0.5 / args.num_steps

        curiosity_loss = args.lambda_1 * (
            (1 - args.beta) * inv_loss + args.beta * forw_loss)
        # ---ICM--->

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss +
            curiosity_loss).backward()  # ICM
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(curiosity.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        ensure_shared_grads(curiosity, shared_curiosity)
        optimizer.step()

    env.close()
