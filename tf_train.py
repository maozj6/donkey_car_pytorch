import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import os


if __name__ == '__main__':
    model = tf.keras.models.Sequential()
    # 多个卷积层
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation="relu",
                                     input_shape=(224, 224, 1)))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    # 将前面卷积层得出的多维数据转为一维
    # 7和前面的kernel_size、padding、MaxPool2D有关
    # Conv2D: 28*28 -> 28*28 (因为padding="same")
    # MaxPool2D: 28*28 -> 14*14
    # Conv2D: 14*14 -> 14*14 (因为padding="same")
    # MaxPool2D: 14*14 -> 7*7
    model.add(tf.keras.layers.Reshape(target_shape=(56 * 56* 64,)))
    # 传入全连接层
    model.add(tf.keras.layers.Dense(1024, activation="relu"))
    model.add(tf.keras.layers.Dense(30, activation="softmax"))

    # compile
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

    train_data=np.load('train.npz')
    x_train=train_data["obs"]
    y_train=train_data['lbl']
    test_data=np.load("test.npz")
    x_test=test_data["obs"]
    y_test=test_data['lbl']

    callbacks = [
        tf.keras.callbacks.EarlyStopping(min_delta=1e-3, patience=5)
    ]

    history = model.fit(x_train, y_train, epochs=15,
                        validation_data=(x_test, y_test),
                        callbacks=callbacks)

    def plot_learning_curves(history):
        pd.DataFrame(history.history).plot(figsize=(8, 5))
        plt.grid(True)
        # plt.gca().set_ylim(0, 1)
        plt.show()


    plot_learning_curves(history)
