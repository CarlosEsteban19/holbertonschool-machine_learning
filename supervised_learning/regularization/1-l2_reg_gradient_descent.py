#!/usr/bin/env python3
"""holberton regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases using gradient descent
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]
        W = weights[f"W{i}"]

        dW = (np.matmul(dZ, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA = np.matmul(W.T, dZ)
            dZ = dA * (1 - np.square(cache[f"A{i-1}"]))

        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db
