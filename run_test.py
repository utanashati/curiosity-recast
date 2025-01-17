import torch
import torch.nn.functional as F

import argparse
import os
import time
from collections import deque

from envs import create_atari_env, create_doom_env, create_picolmaze_env
from model import ActorCritic
from gym import wrappers

import logging
import logger

from os import listdir
from os.path import isfile

import numpy as np

parser = argparse.ArgumentParser(description='Run Test')
parser.add_argument('--base-dir', type=str,
                    help="directory with training runs")
parser.add_argument('--game', type=str, default='atari',
                    help="game (\'atari\' or \'doom\', default: \'atari\')")
parser.add_argument('--env-name', type=str, default='PongDeterministic-v4',
                    help="environment trained on "
                    "(default: PongDeterministic-v4)")
parser.add_argument('--max-episodes', type=int, default=3,
                    help="number of episodes of each test run (default: 3)")
parser.add_argument('--seed', type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument('--record', dest='record', action='store_true',
                    default=False,
                    help="record test runs (default: False).")

parser.add_argument('--num-skip', type=int, default=4,
                    help="number of frames to skip in 'doom' "
                    "(see envs.py, default: 4)")
parser.add_argument('--num-stack', type=int, default=4,
                    help="number of frames to stack in 'doom' "
                    "(see envs.py, default: 4)")

parser.add_argument('--num-rooms', type=int, default=4,
                    help="number of rooms in picolmaze")

if __name__ == '__main__':
    # Parse and check args
    args = parser.parse_args()

    if args.game not in ['atari', 'doom', 'picolmaze']:
        raise ValueError("Choose game between 'doom', 'atari' and 'picolmaze'.")

    if args.game == 'doom':
        args.max_episode_length = 2100
        args.max_episode_length_test = 2100
    elif args.game == 'picolmaze':
        args.max_episode_length = 500
        args.max_episode_length_test = 500
        args.num_stack = 3
    else:
        args.max_episode_length_test = 100
        args.num_stack = 1

    # Create test dir
    test_dir = os.path.join(args.base_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Configure logs
    logger.configure(test_dir, 'test_log.log')

    args_list = [f'{k}: {v}\n' for k, v in vars(args).items()]
    logging.info("\nArguments:\n----------\n" + ''.join(args_list))
    logging.info('Logging run logs to {}'.format(test_dir))

    # Create new dirs
    models_dir = os.path.join(args.base_dir, 'models')
    if not os.path.exists(models_dir):
        raise ValueError(f"No such directory: {models_dir}.")

    if args.record:
        if args.game in ['atari', 'picolmaze']:
            videos_dir = os.path.join(test_dir, 'videos')
            if not os.path.exists(videos_dir):
                logging.info("Created videos dir")
                os.makedirs(videos_dir)
        elif args.game == 'doom':
            recordings_dir = os.path.join(test_dir, 'recordings')
            if not os.path.exists(recordings_dir):
                logging.info("Created recordings dir")
                os.makedirs(recordings_dir)

    model_files = []
    counters = []
    for f in listdir(models_dir):
        if isfile(os.path.join(models_dir, f)) and f.startswith('model'):
            model_files.append(f)
            counters.append(int(f.split('_')[-1].split('.')[0]))

    counter_inds = sorted(range(len(counters)), key=counters.__getitem__)
    model_files = [model_files[i] for i in counter_inds]
    counters = [counters[i] for i in counter_inds]

    rewards = np.zeros((len(counters), args.max_episodes + 1))
    rewards[:, 0] = counters

    if args.game == 'doom':
        env = create_doom_env(
            args.env_name, 0,
            num_skip=args.num_skip, num_stack=args.num_stack)
        if args.record:
            env.set_recordings_dir(recordings_dir)
            logging.info("Set recordings dir")
        env.seed(args.seed + 0)
    elif args.game == 'atari':
        env_to_wrap = create_atari_env(args.env_name)
        env_to_wrap.seed(args.seed + 0)
        env = env_to_wrap
    elif args.game == 'picolmaze':
        env_to_wrap = create_picolmaze_env(args.num_rooms)
        env_to_wrap.seed(args.seed + 0)
        env = env_to_wrap

    env.step(0)

    model = ActorCritic(
        # env.observation_space.shape[0],
        args.num_stack,
        env.action_space)

    model.eval()

    external_reward_sum = 0
    done = True

    count_done = 0

    start_time = time.time()

    passed_time = 0

    # a quick hack to prevent the agent from stucking
    # actions = deque(maxlen=100)
    actions = deque(maxlen=args.max_episode_length_test)
    episode_length = 0

    for i, vals in enumerate(zip(model_files, counters)):
        model_file, current_counter = vals
        count_done = 0

        logging.info(f"File: {model_file}")

        full_model_file = os.path.join(models_dir, model_file)
        model.load_state_dict(torch.load(full_model_file))

        while True:
            episode_length += 1

            # Sync with the shared model
            if done:
                passed_time = time.time() - start_time

                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)

                if args.record:
                    if args.game in ['atari', 'picolmaze']:
                        video_dir = os.path.join(
                            videos_dir,
                            'video_' +
                            time.strftime('%Y.%m.%d-%H.%M.%S_') +
                            str(current_counter))
                        if not os.path.exists(video_dir):
                            os.makedirs(video_dir)
                        logging.info("Created new video dir")
                        env = wrappers.Monitor(env_to_wrap, video_dir, force=False)
                        logging.info("Created new wrapper")
                    elif args.game == 'doom':
                        env.set_current_counter(current_counter)
                        env.set_record()
                        logging.info("Set new recording")

                state = env.reset()
                state = torch.from_numpy(state)
            else:
                cx = cx.detach()
                hx = hx.detach()

            with torch.no_grad():
                value, logit, (hx, cx) = model(state.unsqueeze(0), hx, cx)
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1].flatten().detach()

            state, external_reward, done, _ = env.step(action)
            state = torch.from_numpy(state)

            # external reward = 0 if ICM-only mode
            # external_reward = external_reward * (1 - args.icm_only)
            external_reward_sum += external_reward

            done = done or episode_length >= args.max_episode_length

            # a quick hack to prevent the agent from stucking
            actions.append(action)
            if actions.count(actions[0]) == actions.maxlen:
                done = True

            if done:
                logging.info(
                    "\n\nEp {:3d}: time {}, len {}, total R {:.6f}.\n"
                    "".format(
                        count_done,
                        time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(passed_time)),
                        episode_length, external_reward_sum))

                rewards[i, count_done + 1] = external_reward_sum

                if args.game in ['atari', 'picolmaze']:
                    env.close()  # Close the window after the rendering session
                    env_to_wrap.close()
                logging.info("Episode done, close all")

                episode_length = 0
                external_reward_sum = 0
                actions.clear()

                if count_done >= args.max_episodes - 1:
                    break

                count_done += 1

    np.savetxt(os.path.join(test_dir, 'rewards_{}.csv'.format(
        time.strftime('%Y.%m.%d-%H.%M.%S'))), rewards, delimiter=',')
