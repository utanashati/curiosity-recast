import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Merge Tables')
parser.add_argument('--file-mc', type=str,
                    help="file containing the log of misclassified actions")
parser.add_argument('--file-tb', type=str,
                    help="file containing the log of misclassified actions.")

if __name__ == '__main__':
    args = parser.parse_args()

    mc = np.loadtxt(open(args.file_mc, 'rb'), delimiter=',')
    tb = np.loadtxt(open(args.file_tb, 'rb'), delimiter=',', skiprows=1)

    mc = np.concatenate((tb[:, 1].reshape(-1, 1), mc), axis=1)

    np.savetxt(args.file_mc, mc, delimiter=',')
