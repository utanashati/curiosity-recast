from __future__ import print_function

import argparse
import os
import time

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env, create_doom_env, create_picolmaze_env
from model import ActorCritic, IntrinsicCuriosityModule

from train import train
from test import test
from train_no_curiosity import train_no_curiosity
from test_no_curiosity import test_no_curiosity
from train_curiosity import train_curiosity
from test_curiosity import test_curiosity

from itertools import chain  # ICM

import logging
import logger
import tensorboard_logger as tb
# from torch.utils.tensorboard import SummaryWriter

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# A3C
parser = argparse.ArgumentParser(description='ICM + A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help="learning rate (default: 0.0001)")
parser.add_argument('--gamma', type=float, default=0.99,
                    help="discount factor for rewards (default: 0.99)")
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help="lambda parameter for GAE (default: 1.00)")
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help="value loss coefficient (default: 0.5)")
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help="gradient clipping (default: 50)")
parser.add_argument('--num-steps', type=int, default=20,
                    help="number of forward steps in A3C (default: 20)")
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help="maximum length of an episode (default: 1000000)")
parser.add_argument('--no-shared', dest='no_shared', action='store_true',
                    default=False,
                    help="use an optimizer without shared momentum")

parser.add_argument('--num-skip', type=int, default=4,
                    help="number of frames to skip in 'doom' "
                    "(see envs.py, default: 4)")
parser.add_argument('--num-stack', type=int, default=4,
                    help="number of frames to stack in 'doom' "
                    "(see envs.py, default: 4)")

parser.add_argument('--max-entropy-coef', type=float, default=1.0,
                    help="add nonzero entropy if entropy is less than "
                    "max entropy (default: 1.0)")

# Intrinsic Curiosity Module (ICM)
parser.add_argument('--eta', type=float, default=0.01,
                    help="ICM reward factor (default: 0.01)")
parser.add_argument('--beta', type=float, default=0.2,
                    help="curiosity_loss = (1 - args.beta) * inv_loss + "
                    "args.beta * forw_loss (default: 0.2)")
parser.add_argument('--lambda-1', type=float, default=10,
                    help="the ratio of A3C and ICM learning rates "
                    "(1 / lambda from the paper) (default: 10)")

# General
parser.add_argument('--short-description', default='no-descr',
                    help="short description of the run (used in TensorBoard) "
                    "(default: 'no-descr')")
parser.add_argument('--game', type=str, default='atari',
                    help="game ('atari', 'doom' or 'picolmaze', default: 'atari')")
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help="environment to train on "
                    "(default: PongDeterministic-v4)")
parser.add_argument('--num-processes', type=int, default=4,
                    help="how many training processes to use (default: 4)")
parser.add_argument('--max-episodes', type=int, default=1000,
                    help="finish after _ episodes (default: 1000)")
parser.add_argument('--seed', type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument('--random-seed', dest='random_seed', action='store_true',
                    default=False,
                    help="select random seed [0, 1000] (default: False)")
parser.add_argument('--time-sleep', type=int, default=60,
                    help="sleep time for the test process (default: 60)")
parser.add_argument('--save-model-again-eps', type=int, default=3,
                    help="save the model every _ episodes (default: 3)")
parser.add_argument('--save-video-again-eps', type=int, default=3,
                    help="save the recording every _ episodes (default: 3)")

parser.add_argument('--icm-only', dest='icm_only', action='store_true',
                    default=False,
                    help="train the A3C with ICM rewards only "
                    "(no external rewards) (default: False)")
parser.add_argument('--no-curiosity', dest='no_curiosity', action='store_true',
                    default=False,
                    help="run without curiosity (default: False)")
parser.add_argument('--curiosity-only', dest='curiosity_only',
                    action='store_true', default=False,
                    help="train only curiosity model (frozen A3C, default: False)")

parser.add_argument('--clip', type=float, default=1.0,
                    help="reward clipping value (default: 1.0)")

parser.add_argument('--model-file', type=str, default=None,
                    help="model file to start training with")
parser.add_argument('--curiosity-file', type=str, default=None,
                    help="curiosity file to start training with")
parser.add_argument('--optimizer-file', type=str, default=None,
                    help="optimizer file to start training with")
parser.add_argument('--steps-counter', type=int, default=0,
                    help="set different initial steps counter "
                    "(to continue from trained, default: 0)")

parser.add_argument('--num-rooms', type=int, default=4,
                    help="number of rooms in picolmaze.")


def setup_loggings(args):
    args.sum_base_dir = ('runs/{}/{}({})').format(
        args.env_name, time.strftime('%Y.%m.%d-%H.%M'), args.short_description)

    if not os.path.exists(args.sum_base_dir):
        os.makedirs(args.sum_base_dir)

    logger.configure(args.sum_base_dir, 'rl.log')

    args_list = [f'{k}: {v}\n' for k, v in vars(args).items()]
    logging.info("\nArguments:\n----------\n" + ''.join(args_list))
    logging.info('Logging run logs to {}'.format(args.sum_base_dir))
    tb.configure(args.sum_base_dir)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    # Parse and check args
    args = parser.parse_args()

    if args.game not in ['atari', 'doom', 'picolmaze']:
        raise ValueError("Choose game between 'doom', 'atari' or 'picolmaze'.")

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

    setup_loggings(args)
    # writer = SummaryWriter(args.sum_base_dir)

    if args.random_seed:
        random_seed = torch.randint(0, 1000, (1,))
        logging.info(f"Seed: {int(random_seed)}")
        torch.manual_seed(random_seed)
    else:
        torch.manual_seed(args.seed)

    if args.game == 'doom':
        env = create_doom_env(
            args.env_name, 0,
            num_skip=args.num_skip, num_stack=args.num_stack)
    elif args.game == 'atari':
        env = create_atari_env(args.env_name)
    elif args.game == 'picolmaze':
        env = create_picolmaze_env(args.num_rooms)

    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)
    state = env.reset()
    state = torch.from_numpy(state)

    shared_model = ActorCritic(
        # env.observation_space.shape[0], env.action_space)
        args.num_stack, env.action_space)
    shared_model.share_memory()
    # writer.add_graph(shared_model, (state.unsqueeze(0), hx, cx))

    if not args.no_curiosity:
        # <---ICM---
        shared_curiosity = IntrinsicCuriosityModule(
            # env.observation_space.shape[0], env.action_space)
            args.num_stack, env.action_space)
        shared_curiosity.share_memory()
        # writer.add_graph(
        #     shared_curiosity,
        #     (state.unsqueeze(0), torch.tensor(0).reshape(1, 1), state.unsqueeze(0)))
        # ---ICM--->

    # writer.close()

    if args.no_shared:
        optimizer = None
    else:
        if args.no_curiosity:
            optimizer = my_optim.SharedAdam(
                shared_model.parameters(), lr=args.lr)
        elif not args.no_curiosity:
            if not args.curiosity_only:
                optimizer = my_optim.SharedAdam(  # ICM
                    chain(shared_model.parameters(), shared_curiosity.parameters()),
                    lr=args.lr)
            elif args.curiosity_only:
                optimizer = my_optim.SharedAdam(
                    shared_curiosity.parameters(), lr=args.lr)
        optimizer.share_memory()

    if (args.model_file is not None) and (args.optimizer_file is not None):
        logging.info("Start with a pretrained model")
        shared_model.load_state_dict(torch.load(args.model_file))
        optimizer.load_state_dict(torch.load(args.optimizer_file))
        if (args.curiosity_file is not None):
            if not args.no_curiosity:
                shared_curiosity.load_state_dict(
                    torch.load(args.curiosity_file))
            else:
                raise ValueError(
                    "--curiosity-file is not None but --no-curiosity is chosen. "
                    "Please either set --curiosity-file to None or don't use "
                    "--no-curiosity.")

    if args.curiosity_only:
        if args.model_file is None:
            raise ValueError("Please provide the A3C model file.")
        else:
            shared_model.load_state_dict(torch.load(args.model_file))

    processes = []

    manager = mp.Manager()
    pids = manager.list([])
    train_policy_losses = manager.list([0] * args.num_processes)
    train_value_losses = manager.list([0] * args.num_processes)
    train_rewards = manager.list([0] * args.num_processes)
    counter = mp.Value('i', args.steps_counter)
    lock = mp.Lock()

    if args.no_curiosity:
        logging.info("Train WITHOUT curiosity")
        train_foo = train_no_curiosity
        test_foo = test_no_curiosity
        args_test = (
            0, args, shared_model,
            counter, pids, optimizer, train_policy_losses,
            train_value_losses, train_rewards)
    elif not args.no_curiosity:
        if not args.curiosity_only:
            logging.info("Train WITH curiosity")
            train_foo = train
            test_foo = test
            args_test = (
                0, args, shared_model, shared_curiosity,
                counter, pids, optimizer, train_policy_losses,
                train_value_losses, train_rewards)
        elif args.curiosity_only:
            logging.info("Train curiosity model only (no A3C)")
            train_foo = train_curiosity
            test_foo = test_curiosity
            args_test = (
                0, args, shared_model, shared_curiosity,
                counter, pids, optimizer)

    p = mp.Process(
        target=test_foo, args=args_test)
    p.start()
    processes.append(p)

    for rank in range(1, args.num_processes + 1):
        if args.no_curiosity:
            args_train = (
                rank, args, shared_model,
                counter, lock, pids, optimizer, train_policy_losses,
                train_value_losses, train_rewards)
        elif not args.no_curiosity:
            if not args.curiosity_only:
                args_train = (
                    rank, args, shared_model, shared_curiosity,
                    counter, lock, pids, optimizer, train_policy_losses,
                    train_value_losses, train_rewards)
            elif args.curiosity_only:
                args_train = (
                    rank, args, shared_model, shared_curiosity,
                    counter, lock, pids, optimizer)
        p = mp.Process(
            target=train_foo,
            args=args_train)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
