import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env, create_doom_env
from model import ActorCritic

import tensorboard_logger as tb
import logging
import os
import signal
from gym import wrappers


def test_no_curiosity(
    rank, args, shared_model,
    counter, pids, optimizer, train_policy_losses,
    train_value_losses, train_rewards
):
    models_dir = os.path.join(args.sum_base_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    recordings_dir = os.path.join(args.sum_base_dir, 'recordings')
    if (not os.path.exists(recordings_dir)) and (args.game == 'doom'):
        logging.info("Created recordings dir")
        os.makedirs(recordings_dir)

    videos_dir = args.sum_base_dir + '/videos'
    if (not os.path.exists(videos_dir)) and (args.game == 'atari'):
        os.makedirs(videos_dir)

    torch.manual_seed(args.seed + rank)

    if args.game == 'doom':
        env = create_doom_env(
            args.env_name, rank,
            num_skip=args.num_skip, num_stack=args.num_stack)
        env.set_recordings_dir(recordings_dir)
        logging.info("Set recordings dir")
        env.seed(args.seed + rank)
    elif args.game == 'atari':
        env_to_wrap = create_atari_env(args.env_name)
        env_to_wrap.seed(args.seed + rank)
        env = env_to_wrap

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
    current_counter = 0

    # a quick hack to prevent the agent from stucking
    # actions = deque(maxlen=100)
    actions = deque(maxlen=args.max_episode_length_test)
    episode_length = 0
    while True:
        episode_length += 1

        # Sync with the shared model
        if done:
            passed_time = time.time() - start_time
            current_counter = counter.value

            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)

            if count_done % args.save_video_again_eps == 0:
                if args.game == 'atari':
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
        action = prob.max(1, keepdim=True)[1].detach()

        state, external_reward, done, _ = env.step(action[0, 0].numpy())
        state = torch.from_numpy(state)

        # external reward = 0 if ICM-only mode
        external_reward = external_reward * (1 - args.icm_only)
        external_reward_sum += external_reward

        done = done or episode_length >= args.max_episode_length

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            train_policy_loss_mean = sum(train_policy_losses) / \
                len(train_policy_losses)
            train_value_loss_mean = sum(train_value_losses) / \
                len(train_value_losses)
            train_rewards_mean = sum(train_rewards) / \
                len(train_rewards)
            logging.info(
                "\n\nEp {:3d}: time {}, num steps {}, FPS {:.0f}, len {},\n"
                "        total R {:.6f}, train policy loss {:.6f}, train value loss {:.6f},\n"
                "        train rewards {:.6f}.\n"
                "".format(
                    count_done,
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(passed_time)),
                    current_counter, current_counter / passed_time,
                    episode_length, external_reward_sum,
                    train_policy_loss_mean, train_value_loss_mean,
                    train_rewards_mean))

            if (
                (count_done % args.save_model_again_eps == 0) and
                (optimizer is not None)
            ):
                torch.save(
                    model.state_dict(),
                    models_dir + '/model_' +
                    time.strftime('%Y.%m.%d-%H.%M.%S') +
                    f'_{current_counter}.pth')
                torch.save(
                    optimizer.state_dict(),
                    models_dir + '/optimizer_' +
                    time.strftime('%Y.%m.%d-%H.%M.%S') +
                    f'_{current_counter}.pth')
                logging.info("Saved the model")

            tb.log_value(
                'steps_second', current_counter / passed_time, current_counter)
            tb.log_value('reward', external_reward_sum, current_counter)
            tb.log_value(
                'loss_train_policy_mean', train_policy_loss_mean,
                current_counter)
            tb.log_value(
                'loss_train_value_mean', train_value_loss_mean,
                current_counter)
            tb.log_value(
                'reward_train_mean', train_value_loss_mean,
                current_counter)

            if args.game == 'atari':
                env.close()  # Close the window after the rendering session
                env_to_wrap.close()
            logging.info("Episode done, close all")

            episode_length = 0
            external_reward_sum = 0
            actions.clear()

            if count_done >= args.max_episodes:
                for pid in pids:
                    os.kill(pid, signal.SIGTERM)
                env.close()
                os.kill(os.getpid(), signal.SIGKILL)

            count_done += 1
            time.sleep(args.time_sleep)
