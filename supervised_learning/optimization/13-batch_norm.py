#!/usr/bin/env python3
"""holberton optimization"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    Z_norm = gamma * Z_norm + beta

    return Z_norm
