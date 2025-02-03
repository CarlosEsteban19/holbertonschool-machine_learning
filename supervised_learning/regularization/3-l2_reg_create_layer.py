#!/usr/bin/env python3
"""holberton regularization"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """creates a neural network layer in tensorFlow"""
    tf.random.set_seed(42)

    regularizer = tf.keras.regularizers.L2(lambtha)
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer
    )
    return layer(prev)
