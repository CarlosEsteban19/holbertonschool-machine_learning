#!/usr/bin/env python3
"""holberton regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a layer with L2 regularization
    """
    kernel_initializer = tf.keras.initializers.GlorotUniform()

    dense_layer = tf.keras.layers.Dense(
        n, activation=activation, kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l2(lambtha))

    return dense_layer(prev)
