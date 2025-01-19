#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Tests a neural network"""
    loss, accuracy = network.evaluate(data, labels, verbose=verbose)

    return loss, accuracy
