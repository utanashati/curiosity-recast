import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_atari_env
from model import ActorCritic, IntrinsicCuriosityModule

import tensorboard_logger as tb
import logging
import os
import signal
from gym import wrappers

from model import get_grad_sum


def test(
    rank, args, shared_model, shared_curiosity,
    counter, pids, optimizer
):
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
    curiosity = IntrinsicCuriosityModule(  # ICM
        env_to_wrap.observation_space.shape[0],
        env_to_wrap.action_space)

    model.eval
    curiosity.eval()  # ICM

    external_reward_sum = 0
    curiosity_reward_sum = 0
    curiosity_reward_sum_clipped = 0
    inv_loss = 0
    forw_loss = 0
    curiosity_loss = 0
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
            passed_time = time.time() - start_time
            current_counter = counter.value

            model.load_state_dict(shared_model.state_dict())
            curiosity.load_state_dict(shared_curiosity.state_dict())
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)

            if count_done % args.save_video_again_eps == 0:
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

            state = env.reset()
            state = torch.from_numpy(state)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1, keepdim=True)[1].detach()

        state_old = state  # ICM

        state, external_reward, done, _ = env.step(action[0, 0].numpy())
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
        prob_curiosity = F.softmax(inv_out, dim=-1)
        log_prob_curiosity = F.log_softmax(logit, dim=-1)
        inv_loss += float(-(log_prob_curiosity * prob_curiosity).sum(
            1, keepdim=True).detach())
        forw_loss += float(curiosity_reward.detach())

        curiosity_reward = args.eta * curiosity_reward
        curiosity_reward_sum += curiosity_reward.detach()
        curiosity_reward_sum_clipped += \
            max(min(curiosity_reward.detach(), args.clip), -args.clip)

        external_reward_sum += external_reward
        # ---ICM--->

        done = done or episode_length >= args.max_episode_length

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            # <---ICM---
            inv_loss = inv_loss / episode_length
            forw_loss = forw_loss * (32 * 3 * 3) * 0.5 / episode_length

            curiosity_loss = args.lambda_1 * (
                (1 - args.beta) * inv_loss + args.beta * forw_loss)
            # ---ICM--->

            logging.info(
                "\n\nEp {:3d}: time {}, num steps {}, FPS {:.0f}, len {},\n"
                "        total R {}, curiosity R {:.2f}, curiosity R clipped {:.2f},\n"
                "        inv loss {:.3f}, forw loss {:.3f}, curiosity loss {:.2f}.\n"
                "".format(
                    count_done,
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(passed_time)),
                    current_counter, current_counter / passed_time,
                    episode_length, external_reward_sum, curiosity_reward_sum,
                    curiosity_reward_sum_clipped,
                    inv_loss, forw_loss, curiosity_loss))

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
                    curiosity.state_dict(),
                    models_dir + '/curiosity_' +
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
            tb.log_value('reward_icm', curiosity_reward_sum, current_counter)
            tb.log_value(
                'reward_icm_clipped', curiosity_reward_sum_clipped,
                current_counter)
            tb.log_value('loss_inv', inv_loss, current_counter)
            tb.log_value('loss_forw', forw_loss, current_counter)
            tb.log_value('loss_curiosity', curiosity_loss, current_counter)

            env.close()  # Close the window after the rendering session
            env_to_wrap.close()
            logging.info("Episode done, close all")

            external_reward_sum = 0
            curiosity_reward_sum = 0
            curiosity_reward_sum_clipped = 0
            episode_length = 0
            inv_loss = 0
            forw_loss = 0
            curiosity_loss = 0
            actions.clear()

            if count_done >= args.max_episodes:
                for pid in pids:
                    os.kill(pid, signal.SIGKILL)
                os.kill(os.getpid(), signal.SIGKILL)

            count_done += 1
            time.sleep(args.time_sleep)
