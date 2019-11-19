# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

from os import path, makedirs
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from calciumscoring.datasets import Dataset, read_metadata_file
from calciumscoring.networks import UndilatedConvNet as ConvNet
from calciumscoring.extractors import PatchExtractor
from calciumscoring.resampling import resample_mask, pad_or_crop_image
from calciumscoring.io import read_image, write_image

# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': '2classes',
    'experiment': 'UndilatedDeep65_OTF_FullImage_AllKernels_AllKernelMask',
    'random_seed': 897254,
    'patch_size_mm': 65 * 0.66,
    'patch_size_voxels': 65,
    'minibatch_size': 128,
    'train_data': 'training',
}

# Command line arguments
parser = ArgumentParser()
parser.add_argument('--inputdir', default='/home/user/input')
parser.add_argument('--scratchdir', default='/home/user/scratch')
parser.add_argument('--test_data', default='testing')
parser.add_argument('--restore_epoch', type=int, default=250)
parser.add_argument('--train_scans', type=int, default=1012)
parser.add_argument('--kernels', default='all')  # soft/sharp
parser.add_argument('--stage1', default='combined_DeeplySupervised_FullImage_AllKernels_1012_700')

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Set further directories
config['imagedir'] = path.join(config['scratchdir'], 'images_resampled')
config['overlaydir'] = path.join(config['scratchdir'], 'annotations_resampled')
config['maskdir'] = path.join(config['scratchdir'], 'predictions_{}'.format(config['stage1']), 'calcium_masks')

orgimgdir = path.join(config['scratchdir'], 'images')
if not path.exists(orgimgdir):
    orgimgdir = path.join(config['inputdir'], 'images')
config['original_imagedir'] = orgimgdir

# Initialization
overall_start_time = time()

if config['random_seed'] is not None:
    np.random.seed(config['random_seed'])

# Compile
convnet = ConvNet(config, compile_train_func=False)
print('Successfully compiled convolutional neural network with {} parameters'.format(convnet.count_params()))

# Restore network state
convnet.restore(config['restore_epoch'])
print('Restored network state from epoch {} trained on {} images'.format(config['restore_epoch'], config['train_scans']))

# Create test dataset
metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))
test_data = Dataset(config['test_data'], metadata, config, kernels=config['kernels'])

# Make sure directory for results exists
resultdir = path.join(test_data.resultdir, 'calcium_masks')
if not path.exists(resultdir):
    makedirs(resultdir)

# Classify all candidates
for k, uid in enumerate(sorted(test_data.uids)):
    print('{} ({}/{})'.format(uid, k + 1, len(test_data.uids)))

    image_filename = path.join(test_data.imagedir, uid + '.mha')
    original_image_filename = path.join(config['inputdir'], 'images', uid + '.mha')
    mask_filename = path.join(test_data.maskdir, uid + '.mha')
    output_filename = path.join(resultdir, uid + '.mha')

    if not path.exists(image_filename) or not path.exists(original_image_filename):
        print(' > Image does not exist, skipping...')
        continue

    if not path.exists(mask_filename):
        print(' > Mask does not exist (lung segmentation failed?), skipping...')
        continue

    if path.exists(output_filename):
        print(' > Result file already exists, skipping...')
        continue

    # Load image
    image, spacing, origin = read_image(image_filename)

    patch_extractor = PatchExtractor(
        image, spacing,
        patch_size_voxels=config['patch_size_voxels'],
        patch_size_mm=config['patch_size_mm']
    )
    patch_xy = config['patch_size_voxels']

    # Read mask from first stage -> get non-zero voxels and voxel labels
    mask = read_image(mask_filename, only_data=True)
    sampling_points = np.transpose(np.nonzero(mask))

    # Start measuring the run time from here
    start_time = time()

    # Create empty mask for voxel probabilities
    probmap = np.zeros_like(image, dtype='int16')
    probmap.fill(0)  # needed to actually reserve the memory

    # Classify voxels and write probabilities to map
    for start_batch in range(0, len(sampling_points), config['minibatch_size']):
        end_batch = start_batch + config['minibatch_size']
        batch_of_points = sampling_points[start_batch:end_batch, :]

        # Pass patches through network and obtain posterior probabilities
        patches = np.zeros(shape=(len(batch_of_points), 1, 3, patch_xy, patch_xy), dtype=patch_extractor.dtype)
        for i in range(len(batch_of_points)):
            patches[i, 0, :, :, :] = patch_extractor.extract_orthogonal(batch_of_points[i, :])

        pointwise_probabilities = convnet.classify(patches[:, :, 0, :, :], patches[:, :, 1, :, :], patches[:, :, 2, :, :])

        # Transfer most probable class to map
        predicted_classes = np.argmax(pointwise_probabilities, axis=1)
        for i in range(len(batch_of_points)):
            p = batch_of_points[i]
            probmap[p[0], p[1], p[2]] = int(predicted_classes[i])

    print(' > {}'.format(timedelta(seconds=round(time() - start_time))))

    # Remove negative voxels from stage1 mask
    mask[probmap == 0] = 0

    # Resample mask to original resolution
    original_image, original_spacing, original_origin = read_image(original_image_filename)
    mask = resample_mask(mask, spacing=spacing, new_spacing=original_spacing)
    if mask.shape != original_image.shape:
        mask = pad_or_crop_image(mask, target_shape=original_image.shape, fill=0)
    write_image(output_filename, mask, original_spacing, original_origin)

print('Done with everything, took {} in total'.format(timedelta(seconds=round(time()-overall_start_time)))),
print('({:%d %b %Y %H:%M:%S})'.format(datetime.now()))
