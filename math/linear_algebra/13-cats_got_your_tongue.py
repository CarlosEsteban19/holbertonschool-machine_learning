#!/usr/bin/env python3
"""linear algebra task 13"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices along specified axis"""
    return np.concatenate((mat1, mat2), axis)
