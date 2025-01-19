#!/usr/bin/env python3
"""keras project"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """Trains a model with optional early stopping and learning rate decay"""

    callbacks = []

    if early_stopping and validation_data is not None:
        early_stop_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=verbose,
            restore_best_weights=True
        )
        callbacks.append(early_stop_callback)

    if learning_rate_decay and validation_data is not None:
        def lr_schedule(epoch):
            lr = alpha / (1 + decay_rate * epoch)
            print(f"Epoch {epoch}: LearningRateScheduler setting learning rate to {lr:.10f}.")
            return lr

        lr_scheduler = K.callbacks.LearningRateScheduler(
            lr_schedule, verbose=verbose)
        callbacks.append(lr_scheduler)

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
