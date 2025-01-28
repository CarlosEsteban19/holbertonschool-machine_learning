#!/usr/bin/env python3
"""holberton error analysis"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix.
    """
    recall = sensitivity(confusion)
    prec = precision(confusion)

    f1 = 2 * (prec * recall) / (prec + recall)

    return f1
