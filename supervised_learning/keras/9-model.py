#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def save_model(network, filename):
    """Saves an entire Keras model"""
    network.save(filename)


def load_model(filename):
    """Loads an entire Keras model from a file."""
    model = load_model(filename)
    return model
