import numpy as np
from matplotlib import pyplot as plt

import argparse
import os


parser = argparse.ArgumentParser(description='Plot Test')
parser.add_argument('--file', type=str,
                    help="file containing the log of rewards")
parser.add_argument('--no-curiosity', dest='no_curiosity', action='store_true',
                    default=False,
                    help="plot the model w/o curiosity (only affects the color)")
parser.add_argument('--title', type=str, default='Dense',
                    help="plot title (default: 'Dense').")

if __name__ == '__main__':
    args = parser.parse_args()

    rewards = np.loadtxt(open(args.file, "rb"), delimiter=",")

    with plt.style.context("ggplot"):
        c_i = 1 if args.no_curiosity else 5
        c = list(plt.rcParams['axes.prop_cycle'])[c_i]['color']
        # 1 -- blue
        # 5 -- green

        mean = np.mean(rewards[:, 1:], axis=1)
        std = np.std(rewards[:, 1:], axis=1)
        plt.plot(rewards[:, 0], mean, color=c)

        plt.fill_between(
            rewards[:, 0],
            mean - std, mean + std,
            alpha=0.1, color=c
        )
        plt.plot(rewards[:, 0], mean - std, c=c, linewidth=0.7, alpha=0.3)
        plt.plot(rewards[:, 0], mean + std, c=c, linewidth=0.7, alpha=0.3)

        # plt.xticks(
        #     range(0, int(np.max(rewards[:, 0])), 10**6),
        #     range(0, int(np.max(rewards[:, 0])) // 10**6))
        # plt.xticks(range(0, 18 * 10**6 + 1, 10**6), range(0, 19))
        plt.xticks(range(0, 12 * 10**6 + 1, 10**6), range(0, 13))
        plt.ylim(-0.55, 1.30)
        plt.xlim(-5 * 10**5, 12 * 10**6 + 5 * 10**5)

        plt.xlabel("Number of training steps (in millions)")
        plt.ylabel("Extrinsic Rewards per Episode")
        plt.title(args.title)

        folder = '/'.join(args.file.split('/')[:-1])
        file = args.file.split('/')[-1][:-3]

        plt.savefig(os.path.join(folder, file + 'pdf'), dpi=300)
        plt.close()
