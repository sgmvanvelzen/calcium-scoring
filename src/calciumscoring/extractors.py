# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

from math import floor, ceil
from scipy.ndimage import map_coordinates


floatX = 'float32'


class SliceExtractor:
    dtype = np.dtype(floatX)

    def __init__(self, image):
        self.image = image

    def extract_slice(self, z, axis=0):
        if axis == 0:
            return self.image[z, :, :].astype(self.dtype)
        elif axis == 1:
            return self.image[:, z, :].astype(self.dtype)
        elif axis == 2:
            return self.image[:, :, z].astype(self.dtype)
        else:
            raise ValueError('Invalid axis (should be 0-2)')

    @staticmethod
    def compute_crop(x, axis_length, half_size):
        x_from = x - half_size
        x_to = x + half_size + 1
        padding_left = 0 if x_from >= 0 else abs(x_from)
        padding_right = 0 if x_to < axis_length else x_to - axis_length

        if padding_left > 0:
            x_from = 0

        return x_from, x_to, padding_left, padding_right

    @staticmethod
    def get_crop(image, z, yx, half_size, pad_val):
        y_from, y_to, y_padding_left, y_padding_right = SliceExtractor.compute_crop(yx[0], image.shape[1], half_size)
        x_from, x_to, x_padding_left, x_padding_right = SliceExtractor.compute_crop(yx[1], image.shape[2], half_size)

        if y_padding_left == 0 and y_padding_right == 0 and x_padding_left == 0 and x_padding_right == 0:
            return image[z, y_from:y_to, x_from:x_to].astype(SliceExtractor.dtype)
        else:
            cropped = np.empty(shape=(2 * half_size + 1, 2 * half_size + 1), dtype=SliceExtractor.dtype)
            cropped.fill(pad_val)

            if y_padding_right == 0 and x_padding_right == 0:
                cropped[y_padding_left:, x_padding_left:] = image[z, y_from:y_to, x_from:x_to]
            elif y_padding_right == 0:
                cropped[y_padding_left:, x_padding_left:-x_padding_right] = image[z, y_from:y_to, x_from:x_to]
            elif x_padding_right == 0:
                cropped[y_padding_left:-y_padding_right, x_padding_left:] = image[z, y_from:y_to, x_from:x_to]
            else:
                cropped[y_padding_left:-y_padding_right, x_padding_left:-x_padding_right] = image[z, y_from:y_to, x_from:x_to]
            return cropped

    def extract_cropped_slice(self, z, yx, half_size, pad_val=-1000):
        return SliceExtractor.get_crop(self.image, z, yx, half_size, pad_val)

    def extract_cropped_slab(self, z, yx, half_size, slab_heigth, pad_val=-1000):
        slab = np.empty(shape=(slab_heigth, 2 * half_size + 1, 2 * half_size + 1), dtype=self.dtype)
        slab.fill(pad_val)

        slab_half_height = (slab_heigth - 1) // 2
        slab_start = z - slab_half_height
        for n in range(slab_heigth):
            z = slab_start + n
            if 0 <= z < self.image.shape[0]:
                slab[n, :, :] = self.extract_cropped_slice(z, yx, half_size, pad_val)

        return slab

    def extract_cropped_ortho_slice(self, axis, zyx, half_size, pad_val=-1000):
        image = self.image if axis == 0 else np.swapaxes(self.image, 0, axis)
        z = zyx[axis]
        if axis == 0:
            yx = (zyx[1], zyx[2])
        elif axis == 1:
            yx = (zyx[0], zyx[2])
        else:
            yx = (zyx[1], zyx[0])

        slice = SliceExtractor.get_crop(image, z, yx, half_size, pad_val)
        return slice.T if axis == 2 else slice

    def extract_cropped_ortho_slices(self, z, yx, half_size, pad_val=-1000):
        slices = np.empty(shape=(3, 2 * half_size + 1, 2 * half_size + 1), dtype=SliceExtractor.dtype)
        p = (z,) + yx
        for axis in range(3):
            slices[axis, :, :] = self.extract_cropped_ortho_slice(axis, p, half_size, pad_val)
        return slices


class PatchExtractor:
    dtype = np.dtype(floatX)

    def __init__(self, image, spacing, patch_size_voxels, patch_size_mm, rescale=True):
        self.image = image
        self.patch_size_voxels = patch_size_voxels
        self.rescale = rescale

        image_spacing = spacing
        patch_spacing = float(patch_size_mm) / patch_size_voxels
        self.sample_distances = tuple(patch_spacing / float(image_spacing[i]) * (patch_size_voxels / 2.0) for i in range(3))

    def rescale_patch(self, patch, minval=-800, maxval=1200):
        p = (patch - minval) / float(maxval - minval)
        p[p > 1] = 1.0
        p[p < 0] = 0.0
        return p

    def extract_orthogonal(self, p):
        # Make a list of points where we need to sample the image
        sample_points = [
            np.linspace(p[i] - self.sample_distances[i], p[i] + self.sample_distances[i], self.patch_size_voxels) for i in range(3)
        ]

        patches = np.zeros((3, self.patch_size_voxels, self.patch_size_voxels), dtype=self.dtype)
        for i in range(3):
            if i == 0:
                a = sample_points[0]
                a_min = max(0, int(floor(a[0])) - 1)
                a_max = int(ceil(a[-1])) + 1

                b = sample_points[1]
                b_min = max(0, int(floor(b[0])) - 1)
                b_max = int(ceil(b[-1])) + 1

                slice = self.image[a_min:a_max, b_min:b_max, p[2]]
            elif i == 1:
                a = sample_points[0]
                a_min = max(0, int(floor(a[0])) - 1)
                a_max = int(ceil(a[-1])) + 1

                b = sample_points[2]
                b_min = max(int(floor(b[0])) - 1, 0)
                b_max = int(ceil(b[-1])) + 1

                slice = self.image[a_min:a_max, p[1], b_min:b_max]
            else:
                a = sample_points[1]
                a_min = max(0, int(floor(a[0])) - 1)
                a_max = int(ceil(a[-1])) + 1

                b = sample_points[2]
                b_min = max(0, int(floor(b[0])) - 1)
                b_max = int(ceil(b[-1])) + 1

                slice = self.image[p[0], a_min:a_max, b_min:b_max]

            patches[i, :, :] = map_coordinates(slice, np.meshgrid(a - a_min, b - b_min), order=1, output=self.dtype, cval=-1000).T

            if i < 2:
                # not strictly necessary because the network should learn the same from flipped patches...
                patches[i, :, :] = np.flipud(patches[i, :, :])

            if self.rescale:
                patches[i, :, :] = self.rescale_patch(patches[i, :, :])  # rescale values to [0, 1] using window-leveling

        return patches
