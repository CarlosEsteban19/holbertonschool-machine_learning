#!/usr/bin/env python3
"""holberton regularization"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """dropout gradient decent"""
    m = Y.shape[1]
    dA = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        dZ = dA * (1 - np.tanh(cache['A' + str(i-1)])**2) if i != L else dA
        dW = np.dot(dZ, cache['A' + str(i-1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i != L:
            dA_prev = np.dot(weights['W' + str(i)].T, dZ)
            dA = dA_prev * cache['D' + str(i)] / keep_prob
        else:
            dA = np.dot(weights['W' + str(i)].T, dZ)

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

    return weights
