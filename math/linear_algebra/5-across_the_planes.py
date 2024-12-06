#!/usr/bin/env python3
"""linear algebra task 5"""


def add_matrices2D(mat1, mat2):
    """adds two matrices element wise"""
    if len(mat1[0]) is not len(mat2[0]):
        return None

    new_mat = []
    for i in range(len(mat1)):
        new_arr = []

        for j in range(len(mat1[0])):
            new_arr.append(mat1[i][j] + mat2[i][j])

        new_mat.append(new_arr)
    return new_mat
