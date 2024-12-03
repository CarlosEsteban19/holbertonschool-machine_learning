#!/usr/bin/env python3

def matrix_shape(matrix):
    rows = len(matrix)
    columns = len(matrix[0])

    try:
        depth = len(matrix[0][0])
    except TypeError:
        return [rows] + [columns]

    return [rows] + [columns] + [depth]
