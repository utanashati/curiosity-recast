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


    fig, ax = plt.subplots(side, side)

    for i in range(args.num_rooms):
        img = ax[i % side, i // side].imshow(means[i].reshape(16, 18))
        img.set_cmap('hot')

    plt.savefig(os.path.join(args.folder, 'inv_head_means.pdf'), dpi=300)
    plt.close()


    fig, ax = plt.subplots(side, side)

    for i in range(args.num_rooms):
        img = ax[i % side, i // side].imshow(np.sort(means[i]).reshape(16, 18))
        img.set_cmap('hot')

    plt.savefig(os.path.join(args.folder, 'inv_head_means_sorted.pdf'), dpi=300)
    plt.close()


    fig, ax = plt.subplots(side, side)

    for i in range(args.num_rooms):
        img = ax[i % side, i // side].imshow(stds[i].reshape(16, 18))
        img.set_cmap('hot')

    plt.savefig(os.path.join(args.folder, 'inv_head_stds.pdf'), dpi=300)
    plt.close()


    fig, ax = plt.subplots(side, side)

    for i in range(args.num_rooms):
        img = ax[i % side, i // side].imshow(np.sort(stds[i]).reshape(16, 18))
        img.set_cmap('hot')

    plt.savefig(os.path.join(args.folder, 'inv_head_stds_sorted.pdf'), dpi=300)
    plt.close()


    means_tsne = TSNE(n_components=2).fit_transform(means)
    stds_tsne = TSNE(n_components=2).fit_transform(stds)

    plt.plot(means_tsne[:, 0], means_tsne[:, 1])
    plt.plot(stds_tsne[:, 0], stds_tsne[:, 1])

    plt.savefig(os.path.join(args.folder, 'inv_head_stds_tsne.pdf'), dpi=300)
    plt.close()
