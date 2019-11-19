# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import json
import numpy as np

from time import time
from datetime import datetime, timedelta
from os import path, makedirs
from argparse import ArgumentParser

from calciumscoring.datasets import Dataset, MutableSubsetOfSlices, read_metadata_file
from calciumscoring.datasets import balanced_minibatch_of_cropped_slices_iterator as minibatch_iterator
from calciumscoring.networks import DilatedParallelDeeplySupervisedNetwork as ConvNet

# ----------------------------------------------------------------------------------------------------------------------

# Configuration
config = {
    'model': 'combined',
    'experiment': 'DeeplySupervised_FullImage_AllKernels',
    'random_seed': 148243,
    'slice_size_voxels': 25,
    'slice_padding_voxels': ConvNet.padding,
    'restore_epoch': None,
    'images_per_subset': 60,
    'epochs': 700,
    'batches_per_epoch': 5,
    'batch_size': 10,
    'minibatch_size': 64,
    'lr': 0.0005,
    'lr_decay': 1.0,
    'train_data': 'training',
    'valid_data': 'validation',
    'classes': 7,
}

# Command line arguments
parser = ArgumentParser()
parser.add_argument('--inputdir', default='/home/user/input')
parser.add_argument('--scratchdir', default='/home/user/scratch')
parser.add_argument('--visdom', default=None)
parser.add_argument('--kernels', default='all')  # soft/sharp

# ----------------------------------------------------------------------------------------------------------------------

# Set config values from command line arguments
for k, v in vars(parser.parse_args()).items():
    config[k] = v

# Set further directories
config['imagedir'] = path.join(config['scratchdir'], 'images_resampled')
config['overlaydir'] = path.join(config['scratchdir'], 'annotations_resampled')
config['maskdir'] = path.join(config['scratchdir'], 'masks_stage1_resampled')

# Initialization
overall_start_time = time()

if config['random_seed'] is not None:
    random_seed = config['random_seed']
    if config['restore_epoch'] is not None:
        random_seed *= config['restore_epoch']
    np.random.seed(random_seed)

# Load datasets and turn lists of lesions into lists of voxels
metadata = read_metadata_file(path.join(config['inputdir'], 'dataset.csv'))

train_data = Dataset(config['train_data'], metadata, config, mode='slices', kernels=config['kernels'])
train_subset = MutableSubsetOfSlices(train_data, config['images_per_subset'])

valid_data = Dataset(config['valid_data'], metadata, config, mode='slices', kernels=config['kernels'])
valid_subset = MutableSubsetOfSlices(valid_data, config['images_per_subset'] - 20)

config['train_scans'] = len(train_data)

# Compile network
convnet = ConvNet(config, n_classes=config['classes'], deep_supervision=True)
print('Successfully compiled three convolutional neural network with {} parameters'.format(convnet.count_params()))

if config['restore_epoch']:
    convnet.restore(config['restore_epoch'])
    print('Restored network state from epoch {}'.format(config['restore_epoch']))

# Train the network in epochs
first_epoch = config['restore_epoch'] + 1 if config['restore_epoch'] is not None else 1
learning_rate = config['lr'] * (config['lr_decay']**(first_epoch - 1))
print('Setting initial learning rate to {}'.format(learning_rate))

# Visdom enabled?
if config['visdom'] is None:
    vlc = None
else:
    from calciumscoring.visdom import LearningCurve
    vlc = LearningCurve(env=config['model'], title=config['experiment'], server=config['visdom'])

# Before we start training, lets store the configuration in a file
config_file = path.join(convnet.model_dir, 'settings_from_epoch{}.json'.format(first_epoch))
if not path.exists(convnet.model_dir):
    makedirs(convnet.model_dir)
with open(config_file, 'w') as f:
    json.dump(config, f, sort_keys=True, indent=2, separators=(',', ': '))

# Start training...
for epoch in range(first_epoch, config['epochs'] + 1):
    # Update subset once in a while
    if epoch > first_epoch:
        print('Updating training subset...')
        start_time = time()
        train_subset.swap_images(10)
        valid_subset.swap_images(5)
        print(' > took {}'.format(timedelta(seconds=round(time() - start_time))))

    # Start the actual epoch
    print('Epoch {}/{}'.format(epoch, config['epochs']))
    start_time = time()

    train_performance = []  # 0 = loss, 1 = accuracy (binary), 2 = accuracy (multiclass)
    valid_performance = []

    for n in range(config['batches_per_epoch']):
        # Pass data through network (training)
        for slices, labels in minibatch_iterator(train_subset, n_minibatches=config['batch_size'], axis=None):
            performance = convnet.train(slices, labels, learning_rate)
            train_performance.append(performance)

        # Pass data through network (validation)
        for slices, labels in minibatch_iterator(valid_subset, n_minibatches=config['batch_size'], axis=None):
            performance = convnet.validate(slices, labels)
            valid_performance.append(performance)

        # Push loss to visdom server?
        if vlc is not None:
            progress = epoch - 1 + ((n + 1) / float(config['batches_per_epoch']))
            vlc.post([float(train_performance[-1][0]), float(valid_performance[-1][0])], step=progress)

    # Save network
    convnet.save(epoch)

    # Compute statistics over entire epoch
    train_performance = np.mean(np.asarray(train_performance), axis=0)
    valid_performance = np.mean(np.asarray(valid_performance), axis=0)
    print(' > Train: loss: {}, accuracy: {} (binary) / {} (multiclass) [prior: {}]'.format(*train_performance))
    print(' > Validation: loss: {}, accuracy: {} (binary) / {} (multiclass) [prior: {}]'.format(*valid_performance))

    # Modify learningrate
    if config['lr_decay'] is not None:
        learning_rate *= config['lr_decay']
        print(' > Changed learning rate to {}'.format(learning_rate))

    # Stop time, report runtime for entire epoch
    print(' > took {}'.format(timedelta(seconds=round(time() - start_time))))

print('Done with everything, took {} in total'.format(timedelta(seconds=round(time()-overall_start_time)))),
print('({:%d %b %Y %H:%M:%S})'.format(datetime.now()))
