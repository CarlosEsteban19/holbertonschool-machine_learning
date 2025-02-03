#!/usr/bin/env python3
"""holberton regularization"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """Calculates the cost of a neural network with L2 regularization."""
    l2_losses = tf.convert_to_tensor([loss.numpy() for loss in model.losses])
    return cost + l2_losses
