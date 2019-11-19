# Copyright: (c) 2019, University Medical Center Utrecht 
# GNU General Public License v3.0+ (see LICNSE or https://www.gnu.org/licenses/gpl-3.0.txt)

import numpy as np


def random_item(items):
    return items[np.random.randint(len(items))]
