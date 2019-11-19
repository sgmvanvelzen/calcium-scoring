# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

from os import path, makedirs
from time import time
from datetime import datetime, timedelta
from argparse import ArgumentParser

from calciumscoring.networks import SingleVoxelRemover
from calciumscoring.networks import DilatedParallelDeeplySupervisedNetwork as ConvNet
from calciumscoring.datasets import Dataset, read_metadata_file, classifier_to_overlay_labels
from calciumscoring.io import read_image, write_image
from calciumscoring.extractors import SliceExtractor

# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': 'combined',
    'experiment': 'DeeplySupervised_FullImage_AllKernels',
    'random_seed': 389725,
    'slice_size_voxels': 1,
    'slice_padding_voxels': ConvNet.padding,
    'minibatch_size': 4,
    'train_data': 'training',
    'classes': 7,
    'ortho': True,
}

# Command line arguments
parser = ArgumentParser()
parser.add_argument('--inputdir', default='/home/user/input')
parser.add_argument('--scratchdir', default='/home/user/scratch')
parser.add_argument('--test_data', default='testing')
parser.add_argument('--restore_epoch', type=int, default=700)
parser.add_argument('--train_scans', type=int, default=1012)
parser.add_argument('--kernels', default='all')

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Set further directories
config['imagedir'] = path.join(config['scratchdir'], 'images_resampled')
config['overlaydir'] = path.join(config['scratchdir'], 'annotations_resampled')

# Initialization
overall_start_time = time()

if config['random_seed'] is not None:
    np.random.seed(config['random_seed'])

# Compile network
convnet = ConvNet(config, compile_train_func=False, n_classes=config['classes'])

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
print('Saving to {}'.format(resultdir))

# Classify all candidates
if config['ortho']:
    classify = convnet.classify_ortho
else:
    classify = convnet.classify

remove_single_voxels = SingleVoxelRemover()

for k, uid in enumerate(sorted(test_data.uids)):
    print('{}/{}'.format(k + 1, len(test_data.uids)))

    result_filename = path.join(resultdir, uid + '.mha')
    if path.exists(result_filename):
        print('Result file already exists, skipping...')
        continue

    # Load image
    image_filename = path.join(test_data.imagedir, uid + '.mha')
    if not path.exists(image_filename):
        print('Image file does not exist, skipping...')
        continue

    start_time = time()
    image, spacing, origin = read_image(image_filename)
    slice_extractor = SliceExtractor(image)
    print('  > {} image loading'.format(timedelta(seconds=round(time() - start_time))))

    # Process axial, sagittal and coronal slices to obtain voxelwise probabilities
    n_features = convnet.features_per_orientation
    features = np.zeros(image.shape + (3 * n_features,), dtype='float16')
    features.fill(0)  # needed to actually reserve the memory

    for axis in range(3):
        start_time = time()

        image_shape = image.shape
        if axis == 1:
            image_shape = (image_shape[1], image_shape[0], image_shape[2])
        elif axis == 2:
            image_shape = (image_shape[2], image_shape[0], image_shape[1])

        n_slices = image_shape[0]
        slice_indices = list(range(0, n_slices))
        slice_x = image_shape[1] + config['slice_padding_voxels']
        slice_y = image_shape[2] + config['slice_padding_voxels']
        so = config['slice_padding_voxels'] // 2  # slice offset

        for start_batch in range(0, n_slices, config['minibatch_size']):
            end_batch = start_batch + config['minibatch_size']
            batch_of_slices = slice_indices[start_batch:end_batch]

            slices = np.empty(shape=(len(batch_of_slices), 1, slice_x, slice_y), dtype=slice_extractor.dtype)
            slices.fill(-1000)

            for i in range(len(batch_of_slices)):
                slices[i, 0, so:-so, so:-so] = slice_extractor.extract_slice(batch_of_slices[i], axis=axis)

            f = convnet.extract_features[axis](slices)
            if axis == 0:
                features[start_batch:end_batch, :, :, 0*n_features:1*n_features] = np.transpose(f, (0, 2, 3, 1))
            elif axis == 1:
                features[:, start_batch:end_batch, :, 1*n_features:2*n_features] = np.transpose(f, (2, 0, 3, 1))
            elif axis == 2:
                features[:, :, start_batch:end_batch, 2*n_features:3*n_features] = np.transpose(f, (2, 3, 0, 1))

        print('  > {} classification along axis {}'.format(timedelta(seconds=round(time() - start_time)), axis))

    # Iterate over the slices in batches to turn features into final probabilities
    start_time = time()

    result = np.zeros_like(image, dtype='int16')
    n_slices = image.shape[0]
    slice_indices = list(range(0, n_slices))

    for start_batch in range(0, n_slices, config['minibatch_size']):
        end_batch = start_batch + config['minibatch_size']
        batch_of_slices = slice_indices[start_batch:end_batch]

        batch_probs = classify(np.transpose(features[start_batch:end_batch, :, :, :], (0, 3, 1, 2)))
        result[start_batch:end_batch, :, :] = classifier_to_overlay_labels(np.argmax(batch_probs, axis=1))

    print('  > {} voxel classification'.format(timedelta(seconds=round(time() - start_time))))

    # Remove single voxels and voxels < 130HU
    start_time = time()

    result[image < 130] = 0
    lesions = remove_single_voxels(result)
    result[lesions == 0] = 0

    # Store mask
    write_image(result_filename, result.astype('int16'), spacing, origin)
    print('  > {} post processing and saving mask'.format(timedelta(seconds=round(time() - start_time))))

print('Done with everything, took {} in total'.format(timedelta(seconds=round(time()-overall_start_time)))),
print('({:%d %b %Y %H:%M:%S})'.format(datetime.now()))
