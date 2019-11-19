# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

from os import path
from csv import reader as csvreader
from argparse import ArgumentParser

from calciumscoring.datasets import Dataset, read_metadata_file
from calciumscoring.scores import print_confusion_matrix


def read_cac_scores(filename):
    scores = dict()
    with open(filename) as csvfile:
        for i, row in enumerate(csvreader(csvfile)):
            if i == 0:
                continue  # skip first row = header
            scores[row[1].strip()] = float(row[-4].strip())
    return scores


# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': '2classes',
    'experiment': 'UndilatedDeep65_OTF_FullImage_AllKernels_AllKernelMask',
    'random_seed': 897254,
}

# Command line arguments
parser = ArgumentParser()
parser.add_argument('--inputdir', default='/home/user/input')
parser.add_argument('--scratchdir', default='/home/user/scratch')
parser.add_argument('--test_data', default='testing')
parser.add_argument('--restore_epoch', type=int, default=250)
parser.add_argument('--train_scans', type=int, default=1012)
parser.add_argument('--kernels', default='all')  # soft/sharp

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Initialization
if config['random_seed'] is not None:
    np.random.seed(config['random_seed'])

# Create test dataset
metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))
test_data = Dataset(config['test_data'], metadata, config, kernels=config['kernels'])

# Read scores from CSV files
automatic_scores = read_cac_scores(path.join(test_data.resultdir, 'calcium_scores.csv'))
manual_scores = read_cac_scores(path.join(config['scratchdir'], 'reference_calcium_scores.csv'))

# Discard calcium scores from scans not in the dataset
automatic_scores = dict((k, v) for k, v in automatic_scores.items() if k in test_data.uids)
manual_scores = dict((k, v) for k, v in manual_scores.items() if k in test_data.uids)

print('Dataset size: {}'.format(len(test_data.uids)))
print('Automatic scores: {}'.format(len(automatic_scores)))
print('Reference scores: {}'.format(len(manual_scores)))

# Display confusion matrix
print_confusion_matrix(manual_scores, automatic_scores)
