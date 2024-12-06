#!/usr/bin/env python3
"""linear algebra task 4"""


def add_arrays(arr1, arr2):
    """adds two arrays element wise"""
    if len(arr1) is not len(arr2):
        return None

    new_arr = []
    for i in range(len(arr1)):
        new_arr.append(arr1[i] + arr2[i])

    return new_arr
