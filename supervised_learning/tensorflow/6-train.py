#!/usr/bin/env python3
"""tensorflow project"""
import tensorflow.compat.v1 as tf


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes,
          activations, alpha, iterations, save_path="/tmp/model.ckpt"):
    """Builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(iterations + 1):
            sess.run(train_op, feed_dict={x: X_train, y: Y_train})

            if i % 100 == 0 or i == iterations:
                t_loss, t_acc = sess.run(
                    [loss, accuracy], feed_dict={x: X_train, y: Y_train})
                v_loss, v_acc = sess.run(
                    [loss, accuracy], feed_dict={x: X_valid, y: Y_valid})

                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {t_loss}")
                print(f"\tTraining Accuracy: {t_acc}")
                print(f"\tValidation Cost: {v_loss}")
                print(f"\tValidation Accuracy: {v_acc}")

        save_path = saver.save(sess, save_path)

    return save_path
