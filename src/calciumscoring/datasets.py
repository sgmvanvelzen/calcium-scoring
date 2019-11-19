# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
import pickle
import csv

from os import path
from collections import defaultdict, OrderedDict
from tqdm import tqdm

from .io import read_image
from .extractors import PatchExtractor, SliceExtractor
from .utils import random_item


# Define look up tables for label translation between calcium masks and internal labels
lut_in = {0: 0, 8: 1, 9: 2, 10: 3, 15: 4, 52: 4, 53: 5, 54: 4, 55: 4, 56: 4, 57: 4, 58: 6}
lut_out = {0: 0, 1: 8, 2: 9, 3: 10, 4: 15, 5: 53, 6: 58}

overlay_to_classifier_labels = np.vectorize(lambda v: lut_in[v])
classifier_to_overlay_labels = np.vectorize(lambda v: lut_out[v])

calcium_labels = OrderedDict()
calcium_labels['LAD'] = 8
calcium_labels['LCX'] = 9
calcium_labels['RCA'] = 10
calcium_labels['TAC'] = 15
calcium_labels['AV'] = 53
calcium_labels['MV'] = 58
calcium_labels['CAC'] = [8, 9, 10]
calcium_labels['TOTAL'] = [8, 9, 10, 15, 53, 58]

# Define look up table to distinguish soft and sharp reconstruction kernels
lut_kernels = {
    'STANDARD': 'soft',
    'BONE': 'sharp',
    'LUNG': 'sharp',
    'B30F': 'soft',
    'B50F': 'sharp',
    'B80F': 'sharp',
    'C': 'soft',
    'D': 'sharp',
    'FC10': 'soft',
    'FC51': 'sharp',
}


def read_metadata_file(filename):
    """Reads the dataset.csv file and parses it into a dictionary"""
    metadata = dict()

    with open(filename) as csvfile:
        for i, row in enumerate(csv.reader(csvfile)):
            if i == 0:
                continue  # skip header row

            subset = row[6].strip()
            if subset == 'excluded':
                continue

            metadata[row[1].strip()] = {
                'patient_id': row[0].strip(),
                'series_instance_uid': row[1].strip(),
                'reconstruction_kernel': row[2].strip(),
                'slice_thickness': float(row[3].strip()),  # slice thickness of the original scan
                'slice_order': row[4].strip(),  # slice order of the original DICOM data
                'subset': subset  # training / validation / testing
            }

    return metadata


class Dataset:
    """Represents a subset of the whole dataset, such as the testing set or the training set"""
    def __init__(self, subset, metadata, config, mode='images_only', kernels='all'):
        self.name = subset
        self.kernels = kernels
        self.config = config

        # Make list of UIDs and subset of the metadata
        subsets = set(v['subset'] for v in metadata.values())
        if subset not in subsets:
            raise ValueError('Dataset "{}" does not exist'.format(subset))

        self.metadata = dict()
        for k, v in metadata.items():
            if v['subset'] == subset and (kernels == 'all' or lut_kernels[v['reconstruction_kernel']] == kernels):
                self.metadata[k] = v
        self.uids = frozenset(self.metadata.keys())

        # Define paths
        self.imagedir = config['imagedir'] if 'imagedir' in config else None
        self.overlaydir = config['overlaydir'] if 'overlaydir' in config else None
        self.maskdir = config['maskdir'] if 'maskdir' in config else None

        # Define path for predictions (needs number of scans in dataset)
        n_scans = config['train_scans'] if 'train_scans' in config else len(self.uids)
        resultname = 'predictions_{model}_{experiment}_{n_scans}_{restore_epoch}'.format(n_scans=n_scans, **self.config)
        self.resultdir = path.join(config['scratchdir'], resultname)

        # Extract additional metadata if needed
        if mode == 'images_only':
            pass
        elif mode == 'slices':
            self.slices = self.extract_slices()
        else:
            raise ValueError('Invalid mode "{}"'.format(mode))

    def __len__(self):
        return len(self.uids)

    def extract_slices(self):
        status_msg = 'Extracting list of slices from dataset "{}": '.format(self.name)
        cachename = 'slices_{subset}_{kernels}.pkl'.format(subset=self.name, kernels=self.kernels)
        cachefile = path.join(self.config['scratchdir'], cachename)

        if path.exists(cachefile):
            print(status_msg + 'Reading list of slices from cache...')
            with open(cachefile, 'rb') as f:
                slices = pickle.load(f)
        else:
            slices = defaultdict(list)
            empty_slices = 0
            total_slices = 0

            for uid in tqdm(self.uids, desc='Processing images'):
                overlay = read_image(path.join(self.overlaydir, uid + '.mha'), only_data=True)
                for z in range(overlay.shape[0]):
                    empty_slice = ((overlay[z, :, :] > 0).sum() < 3)
                    slices[uid].append(empty_slice)

                    if empty_slice:
                        empty_slices += 1
                    total_slices += 1
                del overlay

            print('Found {} empty and {} non-empty slices'.format(empty_slices, total_slices - empty_slices))

            # Save to cache
            with open(cachefile, 'wb') as f:
                pickle.dump(slices, f, -1)

        return slices

    def make_patch_extractor(self, uid):
        image_filename = path.join(self.imagedir, uid + '.mha')
        image, spacing, origin = read_image(image_filename)
        return PatchExtractor(
            image, spacing,
            patch_size_voxels=self.config['patch_size_voxels'],
            patch_size_mm=self.config['patch_size_mm']
        )

    def make_slice_extractor(self, uid):
        image_filename = path.join(self.imagedir, uid + '.mha')
        image = read_image(image_filename, only_data=True)
        return SliceExtractor(image)

    def make_overlay_slice_extractor(self, uid):
        overlay_filename = path.join(self.overlaydir, uid + '.mha')
        return SliceExtractor(read_image(overlay_filename, only_data=True))


class MutableSubsetOfSlices:
    """A subset of a dataset which can slide over the entire dataset"""
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size

        # Randomly select N images to form the initial subset
        self.uids = []
        self.slices = [[], []]  # 0 = empty, 1 = not empty
        self.slice_extractors = dict()
        self.overlay_slice_extractors = dict()

        self.unused_uids = list(dataset.uids)
        np.random.shuffle(self.unused_uids)
        self.swap_images(size)

    def __len__(self):
        return len(self.uids)

    def swap_images(self, n_images):
        # Move the first n uids to the end of "unused" and the first n from "unused" to the end of "uids"
        removed_uids = self.uids[:n_images]
        added_uids = self.unused_uids[:n_images]
        self.uids = self.uids[n_images:] + added_uids
        self.unused_uids = self.unused_uids[n_images:] + removed_uids

        np.random.shuffle(self.unused_uids)

        # Prepare slice extractors (holding the image in memory)
        for uid in removed_uids:
            del self.slice_extractors[uid]
            del self.overlay_slice_extractors[uid]

        for uid in added_uids:
            self.slice_extractors[uid] = self.dataset.make_slice_extractor(uid)
            self.overlay_slice_extractors[uid] = self.dataset.make_overlay_slice_extractor(uid)

        # Combine individial lists of slices into two (pos/neg) big lists
        for group in range(2):
            self.slices[group] = []

        for uid in self.uids:
            slice = 0
            for empty_slice in self.dataset.slices[uid]:
                group = 0 if empty_slice else 1
                self.slices[group].append((uid, slice))
                slice += 1

        print(' > Removed {} and added {} images to subset of dataset "{}"'.format(
            len(removed_uids), len(added_uids), self.dataset.name))

        print(' > Subset now contains {} images with {} empty and {} not empty slices'.format(
            len(self.uids), len(self.slices[0]), len(self.slices[1])))


def balanced_minibatch_of_cropped_slices_iterator(subset, n_minibatches, axis=0):
    """ Iterates over a MutableSubsetOfSlices in minibatches """
    data = subset.slices
    slice_xy = subset.dataset.config['slice_size_voxels'] + subset.dataset.config['slice_padding_voxels']
    label_xy = subset.dataset.config['slice_size_voxels']
    minibatch_size = subset.dataset.config['minibatch_size']
    half_padding = subset.dataset.config['slice_padding_voxels'] // 2
    half_slice_width = (subset.dataset.config['slice_size_voxels'] - 1) // 2

    all_slices = data[0] + data[1]

    # Iterate over the data
    axes = (0, 1, 2) if axis is None else (axis,)

    slices = np.zeros(shape=(minibatch_size, 3 if axis is None else 1, slice_xy, slice_xy), dtype=SliceExtractor.dtype)
    labels = np.zeros(shape=(minibatch_size, 3 if axis is None else 1, label_xy, label_xy), dtype='int16')

    for n_processed in range(n_minibatches):
        # Extract patches from slices
        for i in range(minibatch_size):
            if i < minibatch_size // 2:
                # Pick a random point somewhere in any slice
                uid, z = random_item(all_slices)

                image_slice = subset.slice_extractors[uid].extract_slice(z)
                slice_shape = image_slice.shape

                # The first bit of the minibatch should be from locations with high intensities
                if i < minibatch_size // 8:
                    indices = np.transpose(np.where(image_slice >= 130))
                    y, x = tuple(indices[np.random.randint(0, indices.shape[0]), :])
                else:
                    y = np.random.randint(0, slice_shape[0])
                    x = np.random.randint(0, slice_shape[1])
            else:
                # Pick a random positive point and extract a patch around it
                uid, z = random_item(data[1])
                overlay_slice = subset.overlay_slice_extractors[uid].extract_slice(z)
                indices = np.transpose(np.nonzero(overlay_slice))
                y, x = tuple(indices[np.random.randint(0, indices.shape[0]), :])

            p = (z, y, x)
            for axis_index, axis_id in enumerate(axes):
                e = subset.slice_extractors[uid]
                slices[i, axis_index, :, :] = e.extract_cropped_ortho_slice(axis_id, p, half_padding + half_slice_width)

                e = subset.overlay_slice_extractors[uid]
                overlay_cropped = e.extract_cropped_ortho_slice(axis_id, p, half_slice_width, pad_val=0)
                labels[i, axis_index, :, :] = overlay_to_classifier_labels(overlay_cropped)

        yield slices, labels


class MutableSubsetOfPositiveVoxelsWithMask:
    """ A subset of a dataset which can slide over the entire dataset """
    def __init__(self, dataset, size):
        self.dataset = dataset
        self.size = size

        # Randomly select N images to form the initial subset
        self.uids = []
        self.extractors = dict()
        self.positives = []  # contains tuples: (uid, z, y, x, label)
        self.positives_lengths = defaultdict(int)
        self.negatives_in_mask = []  # contains tuples: (uid, z, y, x, label)
        self.negatives_in_mask_lengths = defaultdict(int)

        self.unused_uids = list(dataset.uids)
        np.random.shuffle(self.unused_uids)
        self.swap_images(size)

    def __len__(self):
        return len(self.uids)

    def swap_images(self, n_images):
        if len(self.unused_uids) == 0:
            return

        # Move the first n uids to the end of "unused" and the first n from "unused" to the end of "uids"
        removed_uids = self.uids[:n_images]
        added_uids = self.unused_uids[:n_images]
        self.uids = self.uids[n_images:] + added_uids
        self.unused_uids = self.unused_uids[n_images:] + removed_uids

        np.random.shuffle(self.unused_uids)

        # Prepare slice extractors (holding the image in memory)
        for uid in removed_uids:
            del self.extractors[uid]

        for uid in added_uids:
            self.extractors[uid] = self.dataset.make_patch_extractor(uid)

        # Remove positives and negatives from list
        p_l = 0
        n_l = 0
        for uid in removed_uids:
            p_l += self.positives_lengths[uid]
            n_l += self.negatives_in_mask_lengths[uid]
        self.positives = self.positives[p_l:]
        self.negatives_in_mask = self.negatives_in_mask[n_l:]

        # Extend the list of positives and negatives
        for uid in added_uids:
            overlay = read_image(path.join(self.dataset.overlaydir, uid + '.mha'), only_data=True)
            positive_indices = np.transpose(np.nonzero(overlay))
            l = 0
            for p, q in enumerate(positive_indices):
                label = lut_in[overlay[q[0], q[1], q[2]]]
                if label > 0:
                    self.positives.append((uid, q[0], q[1], q[2], label))
                    l += 1
            self.positives_lengths[uid] = l

            mask = read_image(path.join(self.dataset.maskdir, uid + '.mha'), only_data=True)
            positive_indices = np.transpose(np.nonzero(mask))
            l = 0
            for p, q in enumerate(positive_indices):
                label = overlay[q[0], q[1], q[2]]
                if label == 0:
                    self.negatives_in_mask.append((uid, q[0], q[1], q[2], 0))
                    l += 1
            self.negatives_in_mask_lengths[uid] = l

            del mask, overlay

        print(' > Removed {} and added {} images to subset of dataset "{}"'.format(
            len(removed_uids), len(added_uids), self.dataset.name))

        print(' > Subset now contains {} images with {} positive voxels'.format(
            len(self.uids), len(self.positives)))


def balanced_minibatch_iterator_with_mask(subset, n_minibatches, p_positive=0.5, p_negative_mask=0.4):
    """ Iterates over a MutableSubsetOfPositiveVoxelsWithMask in minibatches """
    positives = subset.positives
    negatives_in_mask = subset.negatives_in_mask

    patch_xy = subset.dataset.config['patch_size_voxels']
    minibatch_size = subset.dataset.config['minibatch_size']

    positives_lut = set((uid, z, y, x) for uid, z, y, x, label in positives)

    # Iterate over the data
    for n_processed in range(n_minibatches):
        minibatch = []

        n_target = int(minibatch_size * p_positive)
        while len(minibatch) < n_target:
            minibatch.append(random_item(positives))

        n_target += int(minibatch_size * p_negative_mask)
        while len(minibatch) < n_target:
            minibatch.append(random_item(negatives_in_mask))

        while len(minibatch) < minibatch_size:
            # Randomly pick a scan and a negative voxel
            uid = random_item(subset.uids)
            image = subset.extractors[uid].image

            while True:
                z = np.random.randint(image.shape[0])
                y = np.random.randint(image.shape[1])
                x = np.random.randint(image.shape[2])

                if image[z, y, x] < 130:
                    continue

                identifier = (uid, z, y, x)
                if identifier in positives_lut:
                    continue

                minibatch.append((uid, z, y, x, 0))
                break

        patches = np.zeros(shape=(minibatch_size, 1, 3, patch_xy, patch_xy), dtype=PatchExtractor.dtype)
        labels = np.zeros(shape=minibatch_size, dtype='int')

        for i in range(minibatch_size):
            uid = minibatch[i][0]
            coords = minibatch[i][1:4]
            labels[i] = minibatch[i][-1]
            patches[i, 0, :, :, :] = subset.extractors[uid].extract_orthogonal(coords)

        yield patches, labels
