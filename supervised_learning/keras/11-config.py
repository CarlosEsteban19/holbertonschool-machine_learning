#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model configuration in JSON format"""
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename, 'r') as f:
        config = f.read()
    model = K.models.model_from_json(config)
    return model
