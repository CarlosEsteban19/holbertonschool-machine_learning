#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network with the Keras library"""
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)

    model.add(K.layers.Dense(
        layers[0], activation=activations[0],
        kernel_regularizer=regularizer, input_shape=(nx,)))
    model.add(K.layers.Dropout(1 - keep_prob))

    for i in range(1, len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=regularizer))
        if i < len(layers) - 1:  # No dropout on the output layer
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
