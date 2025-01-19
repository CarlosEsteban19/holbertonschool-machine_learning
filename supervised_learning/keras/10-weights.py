#!/usr/bin/env python3
"""keras project"""


def save_weights(network, filename, save_format='keras'):
    """Saves the weights of a Keras model to a file"""
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """Loads the weights of a Keras model from a file"""
    network.load_weights(filename)
