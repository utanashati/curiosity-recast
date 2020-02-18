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
parser = argparse.ArgumentParser(description='Forward Predictor')
parser.add_argument('--folder', type=str,
                    help="reference curiosity folder (forward model)")
parser.add_argument('--file', type=str,
                    help="reference curiosity model (forward model)")


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

    forw_out_means = []
    forw_out_stds = []
    for room_i, room in enumerate(env.cpics):
        forw_out_means.append([])
        forw_out_stds.append([])

        env.set_room(room_i)
        prev_rooms = []
        for reverse_action in range(1, 5):
            env.step(reverse_action)
            prev_rooms.append(env.room)

        # 'up', 'right', 'down', 'left'
        actions = {1: 3, 2: 4, 3: 1, 4: 2}
        for i in range(1, 5):
            actions[i] = torch.tensor(actions[i])

        for pic in room:
            for proom_i, prev_room in enumerate(prev_rooms):
                for cpic_i in range(len(env.cpics[prev_room])):
                    state_old = torch.from_numpy(env.cpics[prev_room][cpic_i])
                    state = torch.from_numpy(pic)
                    _, _, forw_out_mean, forw_out_std, _, _, _ = \
                        curiosity(state_old.unsqueeze(0), actions[proom_i + 1], state.unsqueeze(0))
                    forw_out_means[room_i].append(forw_out_mean.detach())
                    forw_out_stds[room_i].append(forw_out_std.detach())

    means = []
    stds = []
    for room_mean, room_std in zip(forw_out_means, forw_out_stds):
        room_mean = torch.cat(room_mean, 0)
        room_std = torch.cat(room_std, 0)
        print(room_mean.shape)
        print(torch.mean(room_mean, dim=0).unsqueeze(0).shape)
        means.append(torch.mean(room_mean, dim=0).unsqueeze(0).numpy())
        stds.append(torch.mean(room_std, dim=0).unsqueeze(0).numpy())

    means = np.concatenate(means, 0)
    stds = np.concatenate(stds, 0)

    np.savetxt(os.path.join(args.folder, 'forw_out_means.csv'), means, delimiter=',')
    np.savetxt(os.path.join(args.folder, 'forw_out_stds.csv'), stds, delimiter=',')

    def process(arr):
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                arr[i][j] = np.insert(arr[i][j].numpy(), 0, i)[np.newaxis, :]

        print(arr[0][0].shape)

        arr = list(itertools.chain(*arr))
        print(len(arr), arr[0].shape)
        return np.concatenate(arr, 0)

    forw_out_means = process(forw_out_means)
    forw_out_stds = process(forw_out_stds)

    np.savetxt(os.path.join(args.folder, 'forw_out_means_arr.csv'), forw_out_means, delimiter=',')
    np.savetxt(os.path.join(args.folder, 'forw_out_stds_arr.csv'), forw_out_stds, delimiter=',')
