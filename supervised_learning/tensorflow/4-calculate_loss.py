#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """Calculates the softmax cross-entropy loss of a prediction"""
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    return loss
