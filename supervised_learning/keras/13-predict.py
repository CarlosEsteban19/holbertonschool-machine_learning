#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Makes a prediction using a neural network"""
    prediction = network.predict(data, verbose=verbose)

    if verbose:
        print(f"Prediction: {prediction}")

    return prediction
