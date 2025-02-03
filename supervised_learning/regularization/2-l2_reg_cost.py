#!/usr/bin/env python3
"""holberton regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization."""
    l2_losses = tf.add_n(model.losses)
    return cost + l2_losses
