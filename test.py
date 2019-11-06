import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic

import tensorboard_logger as tb
import logger
import os
from gym import wrappers


logger = logger.getLogger('test')


def test(rank, args, shared_model, counter, optimizer):
    models_dir = args.sum_base_dir + '/models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    videos_dir = args.sum_base_dir + '/videos'
    if not os.path.exists(videos_dir):
        os.makedirs(videos_dir)

    torch.manual_seed(args.seed + rank)

    env_to_wrap = create_atari_env(args.env_name)
    env_to_wrap.seed(args.seed + rank)

    model = ActorCritic(
        env_to_wrap.observation_space.shape[0],
        env_to_wrap.action_space)

    model.eval()

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

        if done:
            if count_done % args.save_video_again_eps == 0:
                video_dir = os.path.join(videos_dir, time.strftime('%Y.%m.%d-%H.%M.%S_video'))
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)
                print("Created new dir " + video_dir)

                env = wrappers.Monitor(env_to_wrap, video_dir, force=False)
                print("Created new wrapper")

            state = env.reset()
            state = torch.from_numpy(state)

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].numpy()

        state, reward, done, _ = env.step(action[0, 0])
        state = torch.from_numpy(state)
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            passed_time = time.time() - start_time
            logger.info(
                "Episode {}: time {}, num steps {}, FPS {:.0f}, "
                "reward {}, length {}".format(
                    count_done,
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(passed_time)),
                    counter.value, counter.value / passed_time,
                    reward_sum, episode_length))

            if count_done % args.save_model_again_eps == 0:
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

            env.close()  # Close the window after the rendering session
            env_to_wrap.close()
            print("Episode done, close all")

            reward_sum = 0
            episode_length = 0
            actions.clear()
            count_done += 1
            time.sleep(60)
