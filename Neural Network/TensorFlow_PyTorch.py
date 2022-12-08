import random

import matplotlib.pyplot as plt

from functions import *
import os
import tensorflow as tf
from tensorflow import keras

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Using pytorch
# train = open_file('bank-note/train.csv')
# test = open_file('bank-note/test.csv')

# train = data_change_torch(train)
# test = data_change_torch(test)

# torch.manual_seed(42)
# Net = Neural_Net_pytorch(input_shape=5, output_shape=2, hidden_units=5)
# Loss = pytorch_train(Net=Net, train=train)

# Using tensorflow
train = open_file_tf('bank-note/train.csv')
test = open_file_tf('bank-note/test.csv')

train = data_change(train)
test = data_change(test)
train_data = np.array([np.array(data[:-1], ndmin=2) for data in train])
train_label = np.array([data[-1] for data in train])
test_data = np.array([np.array(data[:-1], ndmin=2) for data in test])
test_label = np.array([data[-1] for data in test])

Width = [5, 10, 25, 50, 100]
"""Note or un-note the layer.Dense line to change the depth from 3 to 5 to 9"""
for width in Width:
    model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(width, activation=tf.nn.relu, kernel_initializer='he_normal', bias_initializer='zeros'),
                              keras.layers.Dense(2, activation=tf.nn.softmax)])

    # model = keras.Sequential([keras.layers.Flatten(input_shape=(1, 5)),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(width, activation=tf.nn.tanh, kernel_initializer='glorot_normal', bias_initializer='zeros'),
    #                           keras.layers.Dense(2, activation=tf.nn.softmax)])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    """Start Training"""
    model.fit(train_data, train_label, epochs=20)

    """Testing"""
    train_loss, train_acc = model.evaluate(train_data, train_label)
    test_loss, test_acc = model.evaluate(test_data, test_label)

    print(width, 'training error is', round(100*(1-train_acc), 4), '%')
    print(width, 'testing error is', round(100*(1-test_acc), 4), '%', '\n')
