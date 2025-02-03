#!/usr/bin/env python3
"""holberton regularization"""
import numpy as np
import tensorflow as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation with Dropout
    """
    cache = {}
    A = X
    cache['A0'] = A

    for i in range(1, L+1):
        Z = np.dot(weights['W' + str(i)], A) + weights['b' + str(i)]

        if i == L:
            A = tf.nn.softmax(Z).numpy()
        else:
            A = np.tanh(Z)

        cache['A' + str(i)] = A

        if i != L:
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D
            A /= keep_prob
            cache['D' + str(i)] = D

    return cache
