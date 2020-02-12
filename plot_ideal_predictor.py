import numpy as np
from matplotlib import pyplot as plt

import argparse
import os

from sklearn.manifold import TSNE


parser = argparse.ArgumentParser(description='Plot Test')
parser.add_argument('--folder', type=str,
                    help="folder containing means and stds")
parser.add_argument('--num-rooms', type=int,
                    help="number of rooms in picolmaze")

if __name__ == '__main__':
    args = parser.parse_args()

    means = np.loadtxt(open(
        os.path.join(args.folder, 'inv_head_means.csv'), 'rb'), delimiter=',')
    stds = np.loadtxt(open(
        os.path.join(args.folder, 'inv_head_stds.csv'), 'rb'), delimiter=',')
    stds = np.nan_to_num(stds)

    side = int(args.num_rooms**(1 / 2))

    # vmin_mean = np.min(means)
    # vmax_mean = np.max(means)
    vmin_mean = -2
    vmax_mean = 6

    # vmin_std = np.min(stds)
    # vmax_std = np.max(stds)
    vmin_std = 0
    vmax_std = 3


    fig, axs = plt.subplots(side, side)

    for i in range(args.num_rooms):
        ax = axs[i % side, i // side]
        img = ax.imshow(means[i].reshape(16, 18), vmin=vmin_mean, vmax=vmax_mean)
        img.set_cmap('hot')
        fig.colorbar(img, ax=ax)
        ax.set_axis_off()

    plt.savefig(os.path.join(args.folder, 'inv_head_means.pdf'), dpi=300)
    plt.close()


    fig, axs = plt.subplots(side, side)

    for i in range(args.num_rooms):
        ax = axs[i % side, i // side]
        img = ax.imshow(np.sort(means[i]).reshape(16, 18), vmin=vmin_mean, vmax=vmax_mean)
        img.set_cmap('hot')
        fig.colorbar(img, ax=ax)
        ax.set_axis_off()

    plt.savefig(os.path.join(args.folder, 'inv_head_means_sorted.pdf'), dpi=300)
    plt.close()


    fig, axs = plt.subplots(side, side)

    for i in range(args.num_rooms):
        ax = axs[i % side, i // side]
        img = ax.imshow(stds[i].reshape(16, 18), vmin=vmin_std, vmax=vmax_std)
        img.set_cmap('hot')
        fig.colorbar(img, ax=ax)
        ax.set_axis_off()

    plt.savefig(os.path.join(args.folder, 'inv_head_stds.pdf'), dpi=300)
    plt.close()


    fig, axs = plt.subplots(side, side)

    for i in range(args.num_rooms):
        ax = axs[i % side, i // side]
        img = ax.imshow(np.sort(stds[i]).reshape(16, 18), vmin=vmin_std, vmax=vmax_std)
        img.set_cmap('hot')
        fig.colorbar(img, ax=ax)
        ax.set_axis_off()

    plt.savefig(os.path.join(args.folder, 'inv_head_stds_sorted.pdf'), dpi=300)
    plt.close()


    means_tsne = TSNE(n_components=2).fit_transform(means)
    stds_tsne = TSNE(n_components=2).fit_transform(stds)

    plt.plot(means_tsne[:, 0], means_tsne[:, 1])
    plt.plot(stds_tsne[:, 0], stds_tsne[:, 1])

    plt.savefig(os.path.join(args.folder, 'inv_head_stds_tsne.pdf'), dpi=300)
    plt.close()
