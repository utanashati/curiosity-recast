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

import logging
import logger
import tensorboard_logger as tb
# from torch.utils.tensorboard import SummaryWriter

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# A3C
parser = argparse.ArgumentParser(description='Ideal Predictor')
parser.add_argument('--curiosity-file', type=str, default=None,
                    help="reference curiosity (inverse model)")

parser.add_argument('--num-rooms', type=int, default=4,
                    help="number of rooms in picolmaze")
parser.add_argument('--colors', type=str, default='same_1',
                    help="function that sets up room entropies in picolmaze "
                    "(default: 'same_1').")


if __name__ == '__main__':
    # Parse and check args
    args = parser.parse_args()

    args.game = 'picolmaze'
    args.env_name = 'picolmaze'
    args.max_episode_length = 1000
    args.max_episode_length_test = 1000
    args.num_stack = 3

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

    if args.curiosity_file is not None:
        logging.info("Load curiosity")
        shared_curiosity.load_state_dict(
            torch.load(args.curiosity_file), strict=False)
    else:
        raise ValueError("Curiosity file must be chosen.")
