#!/usr/bin/env python3
"""linear algebra task 7"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates matrices along specified axis"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return [row[:] for row in mat1] + [row[:] for row in mat2]

    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        result = []
        for i in range(len(mat1)):
            result.append(mat1[i] + mat2[i])
        return result

    else:
        return None
