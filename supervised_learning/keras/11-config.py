#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K
import json


def save_config(network, filename):
    """
    Saves a model configuration in JSON format"""
    config = network.get_config()

    with open(filename, 'w') as json_file:
        json.dump(config, json_file)


def load_config(filename):
    """Loads a model with a specific configuration"""
    with open(filename, 'r') as json_file:
        config = json.load(json_file)

    model = K.models.Sequential.from_config(config)

    return model
