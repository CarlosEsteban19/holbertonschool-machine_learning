#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """Creates the training operation for the network using gradient descent"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
