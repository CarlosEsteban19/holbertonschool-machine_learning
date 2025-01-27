#!/usr/bin/env python3
"""holberton optimization"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set with bias correction.
    """
    moving_averages = []
    v = 0
    for t, value in enumerate(data, start=1):
        v = beta * v + (1 - beta) * value
        corrected_v = v / (1 - beta ** t)
        moving_averages.append(corrected_v)
    return moving_averages
