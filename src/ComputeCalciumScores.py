# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

from os import path, makedirs
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from calciumscoring.datasets import Dataset, read_metadata_file, calcium_labels
from calciumscoring.io import read_image
from calciumscoring.scores import compute_calcium_scores

# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': '2classes',
    'experiment': 'UndilatedDeep65_OTF_FullImage_AllKernels_AllKernelMask',
    'random_seed': 897254,
    'min_vol': 1.5,
    'max_vol': 100000000.0,
}

# Command line arguments
parser = ArgumentParser()
parser.add_argument('--inputdir', default='/home/user/input')
parser.add_argument('--scratchdir', default='/home/user/scratch')
parser.add_argument('--test_data', default='testing')
parser.add_argument('--restore_epoch', type=int, default=250)
parser.add_argument('--train_scans', type=int, default=1012)
parser.add_argument('--kernels', default='all')  # soft/sharp
parser.add_argument('--score_annotations', action='store_true', help='Calculate reference scores')
parser.add_argument('--postprocessed', action='store_true', help='Use postprocessed calcium maps')

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Set further directories
imgdir = path.join(config['scratchdir'], 'images')
if not path.exists(imgdir):
    imgdir = path.join(config['inputdir'], 'images')
config['imagedir'] = imgdir
config['overlaydir'] = path.join(config['inputdir'], 'annotations')

# Initialization
overall_start_time = time()

if config['random_seed'] is not None:
    np.random.seed(config['random_seed'])

# Create test dataset
metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))
test_data = Dataset(config['test_data'], metadata, config, kernels=config['kernels'])

# Make sure directory for results exists
if config['score_annotations']:
    result_filename = path.join(config['scratchdir'], 'reference_calcium_scores.csv')
    config['maskdir'] = config['overlaydir']
else:
    resultdir = test_data.resultdir
    if not path.exists(resultdir):
        makedirs(resultdir)

    result_filename = path.join(resultdir, 'calcium_scores.csv')
    if config['postprocessed']:
        config['maskdir'] = path.join(test_data.resultdir, 'calcium_masks_postprocessed')
    else:
        config['maskdir'] = path.join(test_data.resultdir, 'calcium_masks')

test_data.maskdir = config['maskdir']

# Empty file and write header
with open(result_filename, 'w') as csvfile:
    csvfile.write('Patient ID,Series Instance UID')
    for label_name in calcium_labels.keys():
        csvfile.write(',{0} Agatston,{0} Volume'.format(label_name))
    csvfile.write('\n')

# Iterate over all images (UIDs) in dataset
for k, uid in enumerate(sorted(test_data.uids)):
    print('{} ({}/{})'.format(uid, k + 1, len(test_data.uids)))

    # Load image and segmentation result
    image_filename = path.join(test_data.imagedir, uid + '.mha')
    mask_filename = path.join(test_data.maskdir, uid + '.mha')

    if not path.exists(image_filename):
        print(' > Image does not exist, skipping...')
        continue

    if not path.exists(mask_filename):
        print(' > Calcium mask does not exist, skipping...')
        continue

    # Calculate calcium scores
    image, spacing, origin = read_image(image_filename)
    mask = read_image(mask_filename, only_data=True)
    mask[image < 130] = 0

    # Store calcium scores in CSV file
    def calcium_scores(labels):
        return compute_calcium_scores(image, spacing, mask, labels, config['min_vol'], config['max_vol'])

    with open(result_filename, 'a') as csvfile:
        csvfile.write('{},{}'.format(test_data.metadata[uid]['patient_id'], uid))
        for label_name, labels in calcium_labels.items():
            scores = calcium_scores(labels)
            csvfile.write(',{:.3f},{:.3f}'.format(*scores))
            print(' > {}: {:.0f}'.format(label_name, scores[0]))
        csvfile.write('\n')

print('Done with everything, took {} in total'.format(timedelta(seconds=round(time()-overall_start_time)))),
print('({:%d %b %Y %H:%M:%S})'.format(datetime.now()))
