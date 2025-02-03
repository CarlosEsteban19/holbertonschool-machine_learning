#!/usr/bin/env python3
"""holberton regularization"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """stop early"""
    if cost > (opt_cost - threshold):
        count += 1
    else:
        count = 0

    return count >= patience, count
