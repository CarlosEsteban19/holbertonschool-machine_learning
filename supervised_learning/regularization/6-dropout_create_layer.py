#!/usr/bin/env python3
"""holberton regularization"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """dropout create ayer"""
    W = tf.random.normal(
        [n, prev.shape[0]], stddev=tf.sqrt(2. / prev.shape[0]))
    b = tf.zeros([n, 1])
    Z = tf.add(tf.matmul(W, prev), b)

    A = activation(Z)

    if training:
        D = tf.cast(tf.random.uniform(A.shape) < keep_prob, tf.float32)
        A = A * D
        A /= keep_prob
    else:
        D = tf.ones_like(A)

    return A
