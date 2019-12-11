import torch
import torch.nn.functional as F
import torch.optim as optim

from envs import create_atari_env, create_doom_env
from model import ActorCritic

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


def train_no_curiosity(
    rank, args, shared_model,
    counter, lock, pids, optimizer, train_policy_losses,
    train_value_losses, train_rewards
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

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    state = torch.from_numpy(state)
    done = True

    episode_length = 0

    killer = Killer()
    while not killer.kill_now:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
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

        for step in range(args.num_steps):
            if done:
                episode_length = 0
                state = env.reset()
                state = torch.from_numpy(state)
            episode_length += 1

            value, logit, (hx, cx) = model(state.unsqueeze(0),
                                           hx, cx)
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).flatten().detach()
            log_prob = log_prob.gather(1, action.view(1, -1))

            state, reward, done, _ = env.step(action.numpy())
            state = torch.from_numpy(state)

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        train_rewards[rank - 1] = sum(rewards)

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model(state.unsqueeze(0), hx, cx)
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

        train_policy_losses[rank - 1] = float((policy_loss).detach().item())
        train_value_losses[rank - 1] = float((value_loss).detach().item())

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

    env.close()
