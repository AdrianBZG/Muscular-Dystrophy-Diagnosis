#!/usr/bin/env python

# METADATA

__author__ = "Adrian Rodriguez-Bazaga, Josep M. Porta"
__credits__ = ["Adrian Rodriguez-Bazaga", "Josep M. Porta"]
__version__ = "1.0.0"
__email__ = "adrianrodriguezbazaga@gmail.com"

import numpy as np
import collections
from keras.preprocessing import image

def load_image(img_path, target_size=(64,64)):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    return img_tensor

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]