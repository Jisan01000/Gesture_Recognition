"""
data_utils.py
-------------
Loads the Binary Hand Gesture Dataset (Stadelmann, 2011 / stes/gestures on
GitHub, CC BY 4.0) and extracts image features for classical ML models.

Dataset summary
----------------
Source : https://github.com/stes/gestures  (data/gestures_64px.hdf5)
Size   : 9,724 training images + 667 test images, 64x64 px, binary
          (pre-segmented hand silhouette on black background)
Classes: 7 static hand gestures
          0 = fist, 1 = thumb, 2 = pistol, 3 = three,
          4 = four, 5 = five, 6 = peace
License: CC BY 4.0
"""

import h5py
import numpy as np
from skimage.feature import hog

CLASS_NAMES = ["fist", "thumb", "pistol", "three", "four", "five", "peace"]


def load_raw(hdf5_path):
    """Load the raw train/test image arrays and integer labels."""
    with h5py.File(hdf5_path, "r") as f:
        X_train = f["Xt"][:]
        y_train = f["yt"][:]
        X_test = f["Xv"][:]
        y_test = f["yv"][:]
    return X_train, y_train, X_test, y_test


def extract_hog_features(images):
    """
    Convert a batch of 64x64 binary images into HOG (Histogram of Oriented
    Gradients) feature vectors -- a classical computer-vision descriptor
    that captures the edge/shape structure of the hand silhouette, which is
    exactly the information that distinguishes one gesture from another.
    """
    feats = []
    for img in images:
        f = hog(
            img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        feats.append(f)
    return np.array(feats, dtype=np.float32)


def extract_pixel_features(images):
    """Flatten + normalise raw pixels to [0, 1] as a simple baseline feature set."""
    X = images.reshape(len(images), -1).astype(np.float32) / 255.0
    return X
