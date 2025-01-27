#!/usr/bin/env python3
"""holberton optimization"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network.
    """
    dense_layer = tf.keras.layers.Dense(
        n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        use_bias=False
    )

    z = dense_layer(prev)

    gamma = tf.Variable(initial_value=tf.ones((1, n)), trainable=True)
    beta = tf.Variable(initial_value=tf.zeros((1, n)), trainable=True)

    mean, variance = tf.nn.moments(z, axes=[0], keepdims=False)

    z_normalized = (z - mean) / tf.sqrt(variance + 1e-7)

    z_scaled = gamma * z_normalized + beta

    output = activation(z_scaled)

    return output
