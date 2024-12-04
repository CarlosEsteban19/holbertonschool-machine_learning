#!/usr/bin/env python3
"""linear algebra task 3"""


def matrix_transpose(matrix):
    """returns the transpose of a 2D matrix"""
    transposed_matrix = []
    for i in range(len(matrix[0])):
        transposed_row = []
        for row in matrix:
            transposed_row.append(row[i])
        transposed_matrix.append(transposed_row)
    return transposed_matrix
