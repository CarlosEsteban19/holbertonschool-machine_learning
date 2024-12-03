#!/usr/bin/env python3

def matrix_shape(matrix):
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
