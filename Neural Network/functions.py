import numpy as np
import random
import math

import torch
from scipy.optimize import minimize
from torch import nn

def open_file(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    for data in dataset:
        for i in range(len(data)):
            data[i] = float(data[i])
        if data[-1] == 0.0:
            data[-1] = -1
    return dataset

def open_file_tf(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            dataset.append(line.strip().split(','))
    for data in dataset:
        for i in range(len(data)):
            data[i] = float(data[i])
    return dataset

def data_change(data):
    for i in range(len(data)):
        x, y = data[i][:-1], data[i][-1]
        x.append(1)
        x.append(y)
        data[i] = x
    return data

def data_change_torch(data):
    for i in range(len(data)):
        x, y = data[i][:-1], data[i][-1]
        x.append(1)
        x.append(y)
        data[i] = torch.tensor(x)
    return data

def sign(x):
    if x > 0:
        return 1
    else:
        return -1

def learning_rate_update(gamma, t, a):
    return gamma/(1+(gamma/a)*t)

def sigmoid(x):
    return 1/(1+np.e**(-x))

def generate_weights(a, b):
    np.random.seed(5)
    return np.random.normal(0, 1, [a, b])

def generate_weights_0(a, b):
    # np.random.seed(42)
    return np.zeros(shape=[a, b])

class Neural_Net:
    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_layer):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_layer_neural = hidden_layer
        self.weights_initialize()

    def weights_initialize(self):
        """General case"""
        self.weights_0 = generate_weights(self.hidden_layer_neural - 1, self.input_shape)
        self.weights_1 = generate_weights(self.hidden_layer_neural - 1, self.hidden_layer_neural)
        self.weights_2 = generate_weights(self.output_shape, self.hidden_layer_neural)[0]
        """Initial weights equal to all zero"""
        # self.weights_0 = generate_weights_0(self.hidden_layer_neural - 1, self.input_shape)
        # self.weights_1 = generate_weights_0(self.hidden_layer_neural - 1, self.hidden_layer_neural)
        # self.weights_2 = generate_weights_0(self.output_shape, self.hidden_layer_neural)[0]
        """Test case"""
        # self.weights_0 = np.array([[-1, -2, -3], [1, 2, 3]])
        # self.weights_1 = np.array([[-1, -2, -3], [1, 2, 3]])
        # self.weights_2 = np.array([-1, 2, -1.5])

    def forward_pass(self, x):
        x = np.array(x)
        x = np.dot(self.weights_0, x.T)
        x = sigmoid(x)
        x = np.insert(x, 0, 1)
        x = np.dot(self.weights_1, x.T)
        x = sigmoid(x)
        x = np.insert(x, 0, 1)
        x = np.dot(self.weights_2, x.T)
        return sign(x)

    def train(self, input, iter, gamma, a):
        x, y_label = input[:-1], input[-1]
        """Forward pass"""
        input_vector_0 = np.array(x)
        hidden_weights_0 = sigmoid(np.dot(self.weights_0, input_vector_0.T))
        input_vector_1 = np.insert(hidden_weights_0, 0, 1)
        hidden_weights_1 = sigmoid(np.dot(self.weights_1, input_vector_1.T))
        input_vector_2 = np.insert(hidden_weights_1, 0, 1)
        y = np.dot(self.weights_2, input_vector_2.T)
        train_loss = 0.5*(y-y_label)**2
        """Back propagation"""
        loss_derivative = y - y_label
        weight_update_2 = loss_derivative*input_vector_2
        comp_weights_1 = self.weights_2.copy()
        comp_weights_1 = np.delete(comp_weights_1, 0)
        weight_update_1 = np.dot(np.transpose([loss_derivative*comp_weights_1*hidden_weights_1*(1-hidden_weights_1)]), [input_vector_1])
        comp_weights_0 = self.weights_1.copy()
        comp_weights_0 = np.delete(comp_weights_0, 0, axis=1)
        weight_update_0 = np.dot(np.transpose([np.dot(loss_derivative*comp_weights_1*hidden_weights_1*(1-hidden_weights_1), comp_weights_0)*hidden_weights_0*(1-hidden_weights_0)]), [input_vector_0])
        """learning rate update"""
        lr = learning_rate_update(gamma, iter, a)
        print(weight_update_0)
        print(weight_update_1)
        print(weight_update_2, '\n')
        """Update weights"""
        self.weights_0 = self.weights_0 - lr * weight_update_0
        self.weights_1 = self.weights_1 - lr * weight_update_1
        self.weights_2 = self.weights_2 - lr * weight_update_2
        return train_loss

def testing(Net, data):
    error = 0
    for d in data:
        x, y_label = d[:-1], d[-1]
        y = Net.forward_pass(x)
        if y != y_label:
            error += 1
    return round(100*error/len(data), 4)

class Neural_Net_pytorch(nn.Module):
    def __init__(self,
                 input_shape,
                 output_shape,
                 hidden_units):
        super().__init__()

        self.input_layer = nn.Sequential(nn.Linear(in_features=input_shape, out_features=hidden_units),
                                         nn.Tanh())  # the activation function can be modified here, nn.Tanh() or nn.ReLU()
        self.hidden_1 = nn.Sequential(nn.Linear(in_features=hidden_units, out_features=hidden_units),
                                      nn.Tanh())
        self.output_layer = nn.Linear(in_features=hidden_units, out_features=output_shape)

    def forward(self, x):
        x = self.input_layer(x)
        # x = nn.init.xavier_normal_(x)
        x = self.hidden_1(x)
        # x = nn.init.xavier_normal_(x)
        x = self.output_layer(x)
        return x


def pytorch_train(Net, train):
    optimizer = torch.optim.Adam(params=Net.parameters())
    loss_function = nn.MSELoss()
    Loss = []
    epoch = 5
    for i in range(epoch):
        train_set = train.copy()
        random.shuffle(train_set)
        for data in train_set:
            X, y = data[:-1], data[-1]
            Net.train()
            y_pred = Net(X)
            loss = loss_function(y, y_pred.squeeze())
            Loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return Loss
