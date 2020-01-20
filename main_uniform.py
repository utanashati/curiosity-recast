from __future__ import print_function

import argparse
import os
import time

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_picolmaze_env
from model import IntrinsicCuriosityModule2
import colors

from train_uniform import train_uniform
from test_uniform import test_uniform

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
parser.add_argument('--lr', type=float, default=0.001,
                    help="learning rate (default: 0.0001)")
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help="gradient clipping (default: 50)")
parser.add_argument('--num-steps', type=int, default=20,
                    help="number of forward steps in A3C (default: 20)")
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help="maximum length of an episode (default: 1000000)")
parser.add_argument('--no-shared', dest='no_shared', action='store_true',
                    default=False,
                    help="use an optimizer without shared momentum")

# Intrinsic Curiosity Module (ICM)
parser.add_argument('--beta', type=float, default=0.2,
                    help="curiosity_loss = (1 - args.beta) * inv_loss + "
                    "args.beta * forw_loss (default: 0.2)")

# General
parser.add_argument('--short-description', default='no-descr',
                    help="short description of the run (used in TensorBoard) "
                    "(default: 'no-descr')")
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

parser.add_argument('--curiosity-file', type=str, default=None,
                    help="curiosity file to start training with")
parser.add_argument('--optimizer-file', type=str, default=None,
                    help="optimizer file to start training with")
parser.add_argument('--steps-counter', type=int, default=0,
                    help="set different initial steps counter "
                    "(to continue from trained, default: 0)")

parser.add_argument('--num-rooms', type=int, default=4,
                    help="number of rooms in picolmaze")
parser.add_argument('--new-curiosity', dest='new_curiosity', action='store_true',
                    default=False,
                    help="use the new metric of curiosity (default: False)")
parser.add_argument('--colors', type=str, default='same_1',
                    help="function that sets up room entropies in picolmaze "
                    "(default: 'same_1').")


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

    args.game = 'picolmaze'
    args.env_name = 'picolmaze'
    args.max_episode_length = 1000
    args.max_episode_length_test = 1000
    args.num_stack = 3

    setup_loggings(args)
    # writer = SummaryWriter(args.sum_base_dir)

    if args.random_seed:
        random_seed = torch.randint(0, 1000, (1,))
        logging.info(f"Seed: {int(random_seed)}")
        torch.manual_seed(random_seed)
    else:
        torch.manual_seed(args.seed)

    env = create_picolmaze_env(args.num_rooms, getattr(colors, args.colors))

    cx = torch.zeros(1, 256)
    hx = torch.zeros(1, 256)
    state = env.reset()
    state = torch.from_numpy(state)

    # <---ICM---
    shared_curiosity = IntrinsicCuriosityModule2(
        args.num_stack, env.action_space)
    shared_curiosity.share_memory()
    # ---ICM--->

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(
            shared_curiosity.parameters(), lr=args.lr)
        optimizer.share_memory()

    if args.curiosity_file is not None:
        logging.info("Load curiosity")
        shared_curiosity.load_state_dict(
            torch.load(args.curiosity_file), strict=False)

    if args.optimizer_file is not None:
        logging.info("Load optimizer")
        optimizer.load_state_dict(torch.load(args.optimizer_file))

    if args.new_curiosity:
        logging.info("Bayesian curiosity")

    processes = []

    manager = mp.Manager()
    pids = manager.list([])
    train_inv_losses = manager.list([0] * args.num_processes)
    train_forw_losses = manager.list([0] * args.num_processes)
    counter = mp.Value('i', args.steps_counter)
    lock = mp.Lock()

    logging.info("Train curiosity with uniform policy")
    train_foo = train_uniform
    test_foo = test_uniform
    args_test = (
        0, args, shared_curiosity, counter, pids,
        optimizer, train_inv_losses, train_forw_losses)

    p = mp.Process(
        target=test_foo, args=args_test)
    p.start()
    processes.append(p)

    for rank in range(1, args.num_processes + 1):
        args_train = (
            rank, args, shared_curiosity, counter,
            lock, pids, optimizer, train_inv_losses,
            train_forw_losses)
        p = mp.Process(
            target=train_foo,
            args=args_train)
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
