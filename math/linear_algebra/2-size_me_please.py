#!/usr/bin/env python3
"""linear algebra task 2"""

def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    rows = len(matrix)

    try:
        columns = len(matrix[0])
    except TypeError:
        return [rows]

    try:
        depth = len(matrix[0][0])
    except TypeError:
        return [rows] + [columns]

    return [rows] + [columns] + [depth]
