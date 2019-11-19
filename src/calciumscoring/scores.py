# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np

from scipy import ndimage


def compute_calcium_scores(image, spacing, mask, labels, min_vol=None, max_vol=None):
    """Calculates Agatston score and volume for one or multiple labels"""
    binary_mask = np.isin(mask, labels)
    voxel_volume = np.prod(spacing)

    agatston_score = 0
    calcium_volume = 0

    # Find individual lesions (in 3D) so that we can discard too small or too large lesions
    connectivity = ndimage.generate_binary_structure(3, 3)
    lesion_map, n_lesions = ndimage.label(binary_mask, connectivity)

    for lesion in range(1, n_lesions + 1):
        lesion_mask = lesion_map == lesion

        # Ignore too small or too large lesions
        lesion_volume = np.count_nonzero(lesion_mask) * voxel_volume
        if min_vol is not None and lesion_volume < min_vol:
            continue
        if max_vol is not None and lesion_volume > max_vol:
            continue

        calcium_volume += lesion_volume

        # Calculate Agatston score for this lesion
        slices = np.unique(np.nonzero(lesion_mask)[0])
        for z in slices:
            fragment_mask = lesion_mask[z, :, :]
            n_pixels = np.count_nonzero(fragment_mask)

            maximum_intensity = np.max(image[z, :, :][fragment_mask])
            if maximum_intensity < 200:
                coefficient = 1
            elif maximum_intensity < 300:
                coefficient = 2
            elif maximum_intensity < 400:
                coefficient = 3
            else:
                coefficient = 4

            agatston_score += coefficient * n_pixels

    # Change number of pixels into area by multiplying with pixel area + apply correction factor for slice spacing != 3mm
    agatston_score *= spacing[0] / 3.0 * spacing[1] * spacing[2]

    return agatston_score, calcium_volume


def linear_weight_matrix(size):
    j = np.tile(range(size), (size, 1))
    return 1 - np.abs(j.T - j).astype(np.float64) / (size - 1)


def linearly_weighted_kappa(observed):
    observed = observed.astype('float')
    chance_expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    observed_p = observed / observed.sum()
    chance_expected_p = chance_expected / chance_expected.sum()

    w_m = linear_weight_matrix(observed_p.shape[0])

    p_o = np.multiply(w_m, observed_p).sum()
    p_e = np.multiply(w_m, chance_expected_p).sum()

    kappa_w = (p_o - p_e) / (1 - p_e)
    return kappa_w


def agatston_score_to_risk_category(score):
    categories = [10, 100, 1000]  # 0-10, 11-100, 101-1000, >1000

    for category, threshold in enumerate(categories):
        if score <= threshold:
            return category + 1

    return len(categories) + 1


def make_confusion_matrix(ref_scores, auto_scores):
    max_category = agatston_score_to_risk_category(float('inf'))
    m = np.zeros((max_category, max_category), dtype=int)
    for uid in auto_scores:
        if uid in ref_scores:
            c = agatston_score_to_risk_category(auto_scores[uid]) - 1
            r = agatston_score_to_risk_category(ref_scores[uid]) - 1
            m[r, c] += 1
    return m


def print_confusion_matrix(ref_scores, auto_scores):
    m = make_confusion_matrix(ref_scores, auto_scores)

    num_scans = np.sum(m)
    print('{} scan pairs'.format(num_scans))
    print('')

    print('Automatic')
    for r in range(m.shape[0]):
        s = []
        for c in range(m.shape[1]):
            s.append('{}'.format(m[r, c]))
        print('\t'.join(s))
    print('')

    for i in range(len(m)):
        n = np.diagonal(m, i).sum()
        if i > 0:
            n += np.diagonal(m, -i).sum()
        print('{} categories off: {}%'.format(i, n / num_scans * 100))
    print('')

    print('Linearly weighted kappa: {}'.format(linearly_weighted_kappa(m)))
