# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

from __future__ import division

import warnings
import numpy as np

from scipy.ndimage import zoom


def resample_image(image, spacing, new_spacing, order=3):
    resampling_factors = tuple(o / n for o, n in zip(spacing, new_spacing))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return zoom(image, resampling_factors, order=order, mode='nearest')


def resample_mask(mask, spacing, new_spacing):
    return resample_image(mask, spacing, new_spacing, order=0)


def pad_or_crop_image(image, target_shape, fill=-1000):
    new_image = np.ones(shape=target_shape, dtype=image.dtype) * fill

    pads = [0, 0, 0]
    crops = [0, 0, 0]
    for axis in range(3):
        if image.shape[axis] < target_shape[axis]:
            pads[axis] = (target_shape[axis] - image.shape[axis]) // 2
        elif image.shape[axis] > target_shape[axis]:
            crops[axis] = (image.shape[axis] - target_shape[axis]) // 2

    cropped = image[
        crops[0]:crops[0] + target_shape[0],
        crops[1]:crops[1] + target_shape[1],
        crops[2]:crops[2] + target_shape[2]
    ]
    new_image[
        pads[0]:pads[0] + cropped.shape[0],
        pads[1]:pads[1] + cropped.shape[1],
        pads[2]:pads[2] + cropped.shape[2]
    ] = cropped

    return new_image


class WeightedAverageImageResampler:
    def __init__(self, target_slice_thickness, target_slice_spacing):
        self.target_slice_thickness = target_slice_thickness
        self.target_slice_spacing = target_slice_spacing

    def resample(self, image, spacing, origin, slice_thickness):
        slice_spacing = abs(float(spacing[0]))
        slice_thickness = abs(float(slice_thickness))

        # Compute number of slices in resampled image
        scan_thickness = slice_thickness + (slice_spacing * (image.shape[0] - 1))
        target_num_slices = int(np.floor(
            (scan_thickness - self.target_slice_thickness) / self.target_slice_spacing + 1
        ))

        # Compute offset of the origin in the new image
        origin_offset_z = -(0.5 * slice_thickness) + (0.5 * self.target_slice_thickness)

        # Create a new (empty) image volume
        target_shape = (target_num_slices, image.shape[1], image.shape[2])
        resampled_image = np.empty(shape=target_shape, dtype=image.dtype)

        resampled_spacing = (self.target_slice_spacing, spacing[1], spacing[2])
        resampled_origin = (origin[0] + origin_offset_z, origin[1], origin[2])

        # Fill new image with values
        for z in range(resampled_image.shape[0]):
            sum_weights = 0
            sum_values = np.zeros((resampled_image.shape[1], resampled_image.shape[2]), dtype=float)

            slice_begin = z * self.target_slice_spacing
            slice_end = slice_begin + self.target_slice_thickness

            # Find first slice in the old image that overlaps with the new slice
            old_slice = 0
            old_slice_begin = 0
            old_slice_end = slice_thickness

            while old_slice_end < slice_begin:
                old_slice += 1
                old_slice_begin += slice_spacing
                old_slice_end += slice_spacing

            # Find all slices in the old image that overlap with the new slice
            while old_slice < image.shape[0] and old_slice_begin < slice_end:
                if old_slice_end <= slice_end:
                    weight = (old_slice_end - max(slice_begin, old_slice_begin)) / slice_thickness
                    sum_weights += weight
                    sum_values += weight * image[old_slice, :, :]
                elif old_slice_begin >= slice_begin:
                    weight = (slice_end - old_slice_begin) / slice_thickness
                    sum_weights += weight
                    sum_values += weight * image[old_slice, :, :]

                old_slice += 1
                old_slice_begin += slice_spacing
                old_slice_end += slice_spacing

            resampled_image[z, :, :] = np.round(sum_values / sum_weights)

        return resampled_image, resampled_spacing, resampled_origin
