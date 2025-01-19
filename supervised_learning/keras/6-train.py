#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                verbose=True, shuffle=False):
    """Trains a model optional early stopping"""

    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=verbose,
            restore_best_weights=True
        )
        callbacks.append(early_stop_callback)

    history = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )

    return history
