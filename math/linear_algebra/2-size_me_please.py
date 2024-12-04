#!/usr/bin/env python3
"""linear algebra task 2"""


def matrix_shape(matrix):
    """calculates the shape of a matrix"""
    shape = []
    while isinstance(matrix[0], list):
        shape.append(len(matrix))
        matrix = matrix[0]
    shape.append(len(matrix))
    return shape
