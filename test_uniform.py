import time
from collections import deque

import torch
import torch.nn.functional as F

from envs import create_picolmaze_env
from model import IntrinsicCuriosityModule2

import tensorboard_logger as tb
import logging
import os
import signal
from gym import wrappers


def test_uniform(
    rank, args, shared_curiosity, counter, pids,
    optimizer, train_inv_losses, train_forw_losses, env
):
    models_dir = os.path.join(args.sum_base_dir, 'models')
    if not os.path.exists(models_dir):
        logging.info("Created models dir")
        os.makedirs(models_dir)

    open(os.path.join(args.sum_base_dir, 'misclassified.csv'), 'a').close()

    torch.manual_seed(args.seed + rank)

    # env = create_picolmaze_env(args.num_rooms, args.colors, args.periodic)
    env = env.copy()
    env.seed(args.seed + rank)

    env.step(0)

    curiosity = IntrinsicCuriosityModule2(  # ICM
        args.num_stack, env.action_space, args.epsilon)

    curiosity.eval()  # ICM

    inv_loss = torch.tensor(0.0)      # ICM
    forw_loss = torch.tensor(0.0)     # ICM
    curiosity_reward = torch.tensor(0.0)  # ICM
    curiosity_loss = 0  # ICM
    max_softmax = torch.tensor(0.0)
    inv_correct = 0.0
    forw_out_std_mean = torch.tensor(0.0)
    forw_out_mean_mean = torch.tensor(0.0)
    l2_loss_mean = torch.tensor(0.0)
    phi2_mean = torch.tensor(0.0)
    misclassified = [0] * env.action_space.n
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

        if done:
            passed_time = time.time() - start_time
            current_counter = counter.value

            # Sync with the shared model
            curiosity.load_state_dict(shared_curiosity.state_dict())  # ICM
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)

            state = env.reset()
            state = torch.from_numpy(state)
        else:
            cx = cx.detach()
            hx = hx.detach()

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
        current_max_softmax = F.softmax(inv_out, dim=-1).max()
        current_argmax_softmax = F.softmax(inv_out, dim=-1).argmax()
        current_inv_correct = ((action == current_argmax_softmax) * 1).item()
        if not current_inv_correct:
            misclassified[action.item()] += 1

        # In noreward-rl:
        # self.invloss = tf.reduce_mean(
        #     tf.nn.sparse_softmax_cross_entropy_with_logits(logits, aindex),
        #     name="invloss")
        # self.forwardloss = 0.5 * tf.reduce_mean(tf.square(tf.subtract(f, phi2)), name='forwardloss')
        # self.forwardloss = self.forwardloss * 288.0 # lenFeatures=288. Factored out to make hyperparams not depend on it.
        current_inv_loss = F.nll_loss(F.log_softmax(inv_out, dim=-1), action)

        if args.new_curiosity:
            current_forw_loss = bayesian_loss
            if args.add_l2:
                current_forw_loss += l2_loss
        else:
            current_forw_loss = l2_loss
            current_curiosity_reward = l2_loss

        inv_loss += current_inv_loss
        forw_loss += current_forw_loss
        curiosity_reward += current_curiosity_reward
        max_softmax += current_max_softmax
        inv_correct += current_inv_correct
        forw_out_mean_mean += forw_out_mean.mean()
        forw_out_std_mean += forw_out_std.mean()
        l2_loss_mean += l2_loss.mean()
        phi2_mean += phi2.mean()
        # ---ICM--->

        done = done or episode_length >= args.max_episode_length

        # a quick hack to prevent the agent from stucking
        actions.append(action)
        if actions.count(actions[0]) == actions.maxlen:
            done = True

        if done:
            # <---ICM---
            inv_loss = inv_loss / episode_length
            # lenFeatures=288. Factored out to make hyperparams not depend on it.
            # NB: To me, it seems that it only makes hyperparameters DEPEND on it.
            # forw_loss = forw_loss * (32 * 3 * 3) * 0.5 / episode_length
            forw_loss = forw_loss / episode_length
            curiosity_reward = curiosity_reward / episode_length
            max_softmax = max_softmax / episode_length
            inv_correct = inv_correct / episode_length
            forw_out_std_mean = forw_out_std_mean / episode_length
            forw_out_mean_mean = forw_out_mean_mean / episode_length
            l2_loss_mean = l2_loss_mean / episode_length
            phi2_mean = phi2_mean / episode_length

            misclassified = [
                "{:.6f}".format(i / episode_length) for i in misclassified]
            misclassified.insert(0, f"{current_counter}")

            curiosity_loss = (1 - args.beta) * inv_loss + args.beta * forw_loss
            # ---ICM--->

            with open(os.path.join(args.sum_base_dir, 'misclassified.csv'), 'a') as f:
                f.write(','.join(misclassified) + '\n')

            train_inv_loss_mean = sum(train_inv_losses) / \
                len(train_inv_losses)
            train_forw_loss_mean = sum(train_forw_losses) / \
                len(train_forw_losses)
            logging.info(
                "\n\nEp {:3d}: time {}, num steps {}, FPS {:.0f}, len {},\n"
                "        train inv loss {:.5f}, train forw loss {:.5f},\n"
                "        inv loss {:.5f}, forw loss {:.5f}, curiosity loss {:.3f},\n"
                "        l2 loss {:.3f}, curiosity reward {:.3f},\n"
                "        forw out mean {:.3f}, forw out std {:.3f},\n"
                "        phi2 mean {:.3f}, max softmax {:.3f}, inv correct {:.3f}.\n"
                "".format(
                    count_done,
                    time.strftime("%Hh %Mm %Ss",
                                  time.gmtime(passed_time)),
                    current_counter, current_counter / passed_time,
                    episode_length,
                    train_inv_loss_mean, train_forw_loss_mean,
                    inv_loss, forw_loss, curiosity_loss,
                    l2_loss_mean, curiosity_reward,
                    forw_out_mean_mean, forw_out_std_mean,
                    phi2_mean, max_softmax, inv_correct))

            if (
                (count_done % args.save_model_again_eps == 0) and
                (optimizer is not None)
            ):
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
            tb.log_value('loss_inv', inv_loss, current_counter)
            tb.log_value('loss_forw', forw_loss, current_counter)
            tb.log_value('loss_curiosity', curiosity_loss, current_counter)
            tb.log_value('loss_train_inv_mean', train_inv_loss_mean, current_counter)
            tb.log_value('loss_train_forw_mean', train_forw_loss_mean, current_counter)
            tb.log_value('action_max_softmax', max_softmax, current_counter)
            tb.log_value('action_inv_correct', inv_correct, current_counter)
            tb.log_value('reward_icm', curiosity_reward, current_counter)
            # Messed up with these
            # tb.log_value('forw_out_mean_mean', forw_out_std_mean, current_counter)
            # tb.log_value('forw_out_std_mean', forw_out_mean_mean, current_counter)
            # tb.log_value('loss_l2', forw_out_mean_mean, current_counter)
            tb.log_value('forw_out_mean_mean', forw_out_mean_mean, current_counter)
            tb.log_value('forw_out_std_mean', forw_out_std_mean, current_counter)
            tb.log_value('loss_l2', l2_loss_mean, current_counter)
            tb.log_value('phi2_mean', phi2_mean, current_counter)

            env.close()  # Leave or keep??? Close the window after the rendering session
            logging.info("Episode done, close all")

            episode_length = 0
            inv_loss = torch.tensor(0.0)      # ICM
            forw_loss = torch.tensor(0.0)     # ICM
            curiosity_reward = torch.tensor(0.0)  # ICM
            curiosity_loss = 0  # ICM
            max_softmax = torch.tensor(0.0)
            inv_correct = 0.0
            forw_out_std_mean = torch.tensor(0.0)
            forw_out_mean_mean = torch.tensor(0.0)
            l2_loss_mean = torch.tensor(0.0)
            phi2_mean = torch.tensor(0.0)
            misclassified = [0] * env.action_space.n
            actions.clear()

            if count_done >= args.max_episodes:
                for pid in pids:
                    os.kill(pid, signal.SIGTERM)
                env.close()
                os.kill(os.getpid(), signal.SIGKILL)

            count_done += 1
            time.sleep(args.time_sleep)
