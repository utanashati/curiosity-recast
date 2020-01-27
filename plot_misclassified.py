import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

import argparse
import os


parser = argparse.ArgumentParser(description='Plot Misclassified')
parser.add_argument('--file-mc', type=str,
                    help="file containing the log of misclassified actions")
parser.add_argument('--title', type=str, default='Uniform',
                    help="plot title (default: 'Uniform').")

if __name__ == '__main__':
    args = parser.parse_args()

    mc = np.loadtxt(open(args.file_mc, 'rb'), delimiter=',')

    with plt.style.context("ggplot"):
        actions = ['play', 'up', 'right', 'down', 'left']
        for i in range(5):
            if i == 0:
                linewidth = mpl.rcParams['lines.linewidth']
                alpha = 1.0
            else:
                linewidth = mpl.rcParams['lines.linewidth'] * 0.75
                alpha = 0.6
            plt.plot(
                mc[:, 0], mc[:, i + 1], label=actions[i],
                linewidth=linewidth, alpha=alpha)

        plt.legend()
        plt.xlabel("Number of training steps")
        plt.ylabel("Fraction of misclassified actions")
        plt.title(args.title)
        plt.ylim(-0.01, 0.21)

        folder = '/'.join(args.file_mc.split('/')[:-1])
        file = args.file_mc.split('/')[-1][:-3]

        plt.savefig(os.path.join(folder, file + 'pdf'), dpi=300)
        plt.close()
