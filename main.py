from __future__ import print_function

import argparse
import os
import time

import torch
import torch.multiprocessing as mp

import my_optim
from envs import create_atari_env, create_doom_env
from model import ActorCritic, IntrinsicCuriosityModule
from test import test
from test_no_curiosity import test_no_curiosity
from train import train
from train_lock import train_lock
from train_no_curiosity import train_no_curiosity

from itertools import chain  # ICM

import tensorboard_logger as tb
import logging
import logger

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4,
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=1000000,
                    help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='PongDeterministic-v4',
                    help='environment to train on (default: PongDeterministic-v4)')
parser.add_argument('--no-shared', dest='no_shared', action='store_true', default=False,
                    help='use an optimizer without shared momentum')

parser.add_argument('--short-description', default='no_descr',
                    help='short description of the run params '
                    '(used in TensorBoard)')
parser.add_argument('--save-model-again-eps', type=int, default=3,
                    help='save the model every _ episodes')
parser.add_argument('--save-video-again-eps', type=int, default=3,
                    help='save the recording every _ episodes')
parser.add_argument('--time-sleep', type=int, default=60,
                    help='sleep time for test.py')
parser.add_argument('--lock', dest='lock', action='store_true', default=False,
                    help='whether to lock gradient update in train.py')
parser.add_argument('--clip', type=float, default=1.0,
                    help='reward clipping value')
parser.add_argument('--icm-only', dest='icm_only', action='store_true', default=False,
                    help='ICM only (no external reward)')
parser.add_argument('--eta', type=float, default=0.01,
                    help='ICM reward factor')
parser.add_argument('--beta', type=float, default=0.2,
                    help='curiosity_loss = (1 - args.beta) * inv_loss + args.beta * forw_loss')
parser.add_argument('--lambda-1', type=float, default=10,
                    help='1 / lambda from the paper')
parser.add_argument('--max-episodes', type=int, default=1000,
                    help='finish after _ episodes')
parser.add_argument('--random-seed', dest='random_seed', action='store_true', default=False,
                    help='random seed [0, 1000]')
parser.add_argument('--no-curiosity', dest='no_curiosity', action='store_true', default=False,
                    help='run without curiosity')
parser.add_argument('--game', type=str, default='atari',
                    help='game mode')

parser.add_argument('--model-file', type=str, default=None)
parser.add_argument('--curiosity-file', type=str, default=None)
parser.add_argument('--optimizer-file', type=str, default=None)


def setup_loggings(args):
    # current_path = os.path.dirname(os.path.realpath(__file__))
    # args.sum_base_dir = (current_path + '/runs/{}/{}({})').format(
    args.sum_base_dir = ('runs/{}/{}({})').format(
        args.env_name, time.strftime('%Y.%m.%d-%H.%M'), args.short_description)

    if not os.path.exists(args.sum_base_dir):
        os.makedirs(args.sum_base_dir)

    logger.configure(args.sum_base_dir)

    args_list = [f'{k}: {v}\n' for k, v in vars(args).items()]
    logging.info("\nArguments:\n----------\n" + ''.join(args_list))
    logging.info('Logging run logs to {}'.format(args.sum_base_dir))
    tb.configure(args.sum_base_dir)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    setup_loggings(args)

    if args.random_seed:
        random_seed = torch.randint(0, 1000, (1,))
        logging.info(f"Seed: {int(random_seed)}")
        torch.manual_seed(random_seed)
    else:
        torch.manual_seed(args.seed)

    if args.game == 'doom':
        env = create_doom_env(args.env_name, 0)
    elif args.game == 'atari':
        env = create_atari_env(args.env_name)
    else:
        raise ValueError("Choose game between 'doom' and 'atari'.")

    shared_model = ActorCritic(
        env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    #<---ICM---
    shared_curiosity = IntrinsicCuriosityModule(
        env.observation_space.shape[0], env.action_space)
    shared_curiosity.share_memory()
    #---ICM--->

    if args.no_shared:
        optimizer = None
    else:
        # optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer = my_optim.SharedAdam(  # ICM
            chain(shared_model.parameters(), shared_curiosity.parameters()),
            lr=args.lr)
        optimizer.share_memory()

    if (args.model_file is not None) and \
        (args.curiosity_file is not None) and \
        (args.optimizer_file is not None):
        logging.info("Start with a pretrained model")
        shared_model.load_state_dict(torch.load(args.model_file))
        shared_curiosity.load_state_dict(torch.load(args.curiosity_file))
        optimizer.load_state_dict(torch.load(args.optimizer_file))

    processes = []

    manager = mp.Manager()
    pids = manager.list([])
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    train_foo = train
    test_foo = test
    args_test = (
        0, args, shared_model, shared_curiosity,
        counter, pids, optimizer)

    if args.lock:
        train_foo = train_lock
    elif args.no_curiosity:
        logging.info("Train without curiosity")
        train_foo = train_no_curiosity
        test_foo = test_no_curiosity
        args_test = (
            0, args, shared_model,
            counter, pids, optimizer)

    p = mp.Process(
        target=test_foo, args=args_test)
    p.start()
    processes.append(p)

    for rank in range(1, args.num_processes + 1):
        if args.no_curiosity:
            args_train = (
                rank, args, shared_model,
                counter, lock, pids, optimizer)
        else:
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
