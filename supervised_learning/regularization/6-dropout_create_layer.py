#!/usr/bin/env python3
"""holberton regularization"""
import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """dropout create ayer"""
    m = prev.shape[1]
    W = np.random.randn(n, prev.shape[0]) * np.sqrt(2. / prev.shape[0])
    b = np.zeros((n, 1))
    Z = np.dot(W, prev) + b

    A = activation(Z)

    if training:
        D = np.random.rand(*A.shape) < keep_prob
        A = A * D
        A /= keep_prob
    else:
        D = np.ones_like(A)

    return A
