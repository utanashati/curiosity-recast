import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import tensorboard_logger as tb
import logger
import os


logger = logger.getLogger('test')


def test(rank, args, shared_model, counter, optimizer):
    models_dir = args.sum_base_dir + '/models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    model.eval()

    state = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True

    count_done = 0

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            count_done += 1

            passed_time = time.time() - start_time
            logger.info(
                "Episode {}: time {}, num steps {}, FPS {:.0f}, "
                "reward {}, length {}".format(
                    count_done,
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(passed_time)),
                    counter.value, counter.value / passed_time,
                    reward_sum, episode_length))

            if count_done % args.save_again_eps == 0:
                torch.save(
                    model.state_dict(),
                    models_dir + '/model_' +
                    time.strftime('%Y.%m.%d-%H.%M.%S') + '.pth')
                torch.save(
                    optimizer.state_dict(),
                    models_dir + '/optimizer_' +
                    time.strftime('%Y.%m.%d-%H.%M.%S') + '.pth')
                print("Saved the model")

            tb.log_value(
                'steps_second', counter.value / passed_time, counter.value)
            tb.log_value('reward', reward_sum, counter.value)

            reward_sum = 0
            episode_length = 0
            actions.clear()
            state = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)
