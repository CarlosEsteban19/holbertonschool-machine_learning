#!/usr/bin/env python3
"""holberton error analysis"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix.
    """
    classes = labels.shape[1]

    confusion = np.zeros((classes, classes), dtype=int)

    for true_label, predicted_label in zip(labels, logits):
        true_class = np.argmax(true_label)
        predicted_class = np.argmax(predicted_label)

        confusion[true_class, predicted_class] += 1

    return confusion
