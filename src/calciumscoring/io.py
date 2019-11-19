# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np
import SimpleITK as sitk

from scipy.ndimage import label as label_connected_components
from collections import OrderedDict, defaultdict


try:
    import image_loader  # custom library for reading (enhanced) DICOM volumes
except ImportError:
    image_loader = None
    print('WARNING (calciumscoring.io): image_loader not available, cannot read DICOM images')


def read_image(filename, only_data=False):
    """ Reads an mhd file and returns it as a numpy array -> order is (z, y, x) !!! """
    image = sitk.ReadImage(filename)
    data = sitk.GetArrayFromImage(image)

    if only_data:
        return data

    spacing = tuple(reversed(image.GetSpacing()))
    origin = tuple(reversed(image.GetOrigin()))
    return data, spacing, origin


def read_dicom_image(filename, only_data=False):
    if image_loader is None:
        raise RuntimeError('Cannot read DICOM image because image_loader library is not installed')

    data = image_loader.load_dicom_image(filename)
    if only_data:
        return data[0]
    else:
        return data[0], data[3], data[2]


def write_image(filename, array, spacing=None, origin=None):
    """ Writes a numpy array into a file (mhd typically) """
    image = sitk.GetImageFromArray(array)
    if spacing is not None:
        image.SetSpacing(tuple(reversed(spacing)))
    if origin is not None:
        image.SetOrigin(tuple(reversed(origin)))
    sitk.WriteImage(image, filename, True)


def volume_iterator(image):
    for z in range(image.shape[0]):
        for y in range(image.shape[1]):
            for x in range(image.shape[2]):
                yield (z, y, x), image[z, y, x]


def extract_calcium_candidates_from_mask(mask_tuple_or_filename, min_vol=1.5, max_vol=5000.0):
    """
    Returns a dictionary where the keys are tuples,
    which are the coordinates of one voxel in the lesion,
    and the values are the centers of the lesions
    """
    if isinstance(mask_tuple_or_filename, str):
        mask, spacing, origin = read_image(mask_tuple_or_filename)
    else:
        mask, spacing, origin = mask_tuple_or_filename

    # Connected-component analysis with 26-connectivity
    label_volume, num_labels = label_connected_components(mask > 0, np.ones((3, 3, 3)))

    # Discard those that are too small or too large
    volumes = dict((label + 1, 0.0) for label in range(num_labels))
    volume_per_voxel = spacing[0] * spacing[1] * spacing[2]
    for l in np.nditer(label_volume):
        label = l.item()
        if label > 0:
            volumes[label] += volume_per_voxel

    candidates = set()
    for label, volume in volumes.items():
        if min_vol <= volume <= max_vol:
            candidates.add(label)

    for l in np.nditer(label_volume, op_flags=['readwrite']):
        label = l.item()
        if label > 0 and label not in candidates:
            l[...] = 0

    # Make list of all voxels per lesion
    voxels = defaultdict(list)
    for z in range(label_volume.shape[0]):
        for y in range(label_volume.shape[1]):
            for x in range(label_volume.shape[2]):
                label = label_volume[z, y, x]
                if label > 0:
                    voxels[label].append((z, y, x))

    # Make list of centers + all voxels inside the lesion
    lesions = OrderedDict()
    for label in voxels:
        identifier = voxels[label][0]
        for i in range(1, len(voxels[label])):
            voxel = voxels[label][i]
            if voxel[0] > identifier[0]:
                identifier = voxel
            elif voxel[0] == identifier[0]:
                if voxel[1] > identifier[1]:
                    identifier = voxel
                elif voxel[1] == identifier[1]:
                    if voxel[2] > identifier[2]:
                        identifier = voxel

        center = np.mean(voxels[label], axis=0)
        for i in range(3):
            center[i] = origin[i] + center[i] * spacing[i]

        volume = float(len(voxels[label])) * volume_per_voxel

        identifier = '{}-{}-{}'.format(int(identifier[2]), int(identifier[1]), int(identifier[0]))  # x-y-z
        lesions[identifier] = (voxels[label], center, volume)

    return lesions
