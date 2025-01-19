#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """Builds a neural network using the functional API of Keras"""
    inputs = K.Input(shape=(nx,))

    regularizer = K.regularizers.L2(lambtha)

    x = K.layers.Dense(layers[0], activation=activations[0],
                       kernel_regularizer=regularizer)(inputs)
    x = K.layers.Dropout(1 - keep_prob)(x)

    for i in range(1, len(layers)):
        x = K.layers.Dense(layers[i], activation=activations[i],
                           kernel_regularizer=regularizer)(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
