from __future__ import print_function

import argparse
import os
import torch

from model import IntrinsicCuriosityModule2

import pickle as pkl
import numpy as np
import itertools


# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
# A3C
parser = argparse.ArgumentParser(description='Ideal Predictor')
parser.add_argument('--folder', type=str,
                    help="reference curiosity folder (inverse model)")
parser.add_argument('--file', type=str,
                    help="reference curiosity model (inverse model)")


if __name__ == '__main__':
    # Parse and check args
    args = parser.parse_args()

    args.game = 'picolmaze'
    args.env_name = 'picolmaze'
    args.max_episode_length = 1000
    args.max_episode_length_test = 1000
    args.num_stack = 3

    env = pkl.load(open(os.path.join(args.folder, 'env.pkl'), 'rb'))

    curiosity = IntrinsicCuriosityModule2(args.num_stack, env.action_space)
    curiosity.load_state_dict(torch.load(args.file), strict=False)

    action = torch.tensor(0)

    phis = []
    for i, room in enumerate(env.cpics):
        phis.append([])
        for pic in room:
            state = torch.from_numpy(pic)
            _, phi, _, _, _, _, _ = \
                curiosity(state.unsqueeze(0), action, state.unsqueeze(0))
            phis[i].append(phi.detach())

    # means = []
    # stds = []
    # for room in phis:
    #     room = torch.cat(room, 0)
    #     print(room.shape)
    #     print(torch.mean(room, dim=0).unsqueeze(0).shape)
    #     means.append(torch.mean(room, dim=0).unsqueeze(0).numpy())
    #     stds.append(torch.std(room, dim=0).unsqueeze(0).numpy())

    # means = np.concatenate(means, 0)
    # stds = np.concatenate(stds, 0)

    # np.savetxt(os.path.join(args.folder, 'inv_head_means.csv'), means, delimiter=',')
    # np.savetxt(os.path.join(args.folder, 'inv_head_stds.csv'), stds, delimiter=',')

    for i in range(len(phis)):
        for j in range(len(phis[i])):
            phis[i][j] = np.insert(phis[i][j].numpy(), 0, i)[np.newaxis, :]

    print(phis[0][0].shape)

    phis = list(itertools.chain(*phis))
    print(len(phis), phis[0].shape)
    phis = np.concatenate(phis, 0)
    np.savetxt(os.path.join(args.folder, 'inv_head_phis.csv'), phis, delimiter=',')
