#!/usr/bin/env python3
"""keras project"""
import numpy as np


def one_hot(labels, classes=None):
    """Converts a label vector into a one-hot matrix"""
    if classes is None:
        classes = np.max(labels) + 1

    one_hot_matrix = np.zeros((labels.shape[0], classes))
    one_hot_matrix[np.arange(labels.shape[0]), labels] = 1

    return one_hot_matrix
