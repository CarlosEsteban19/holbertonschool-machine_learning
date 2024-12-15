#!/usr/bin/env python3
"""calculus task 9"""


def summation_i_squared(n):
    """sumatoria al cuadrado"""
    if not isinstance(n, int) or n < 1:
        return None

    # Base case / turn around you recursive soab
    if n == 1:
        return 1

    # Recursioooooon (not a loop :D)
    return n**2 + summation_i_squared(n - 1)
