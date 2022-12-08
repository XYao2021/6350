import matplotlib.pyplot as plt

from functions import *
import numpy as np
import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
train = open_file('bank-note/train.csv')
test = open_file('bank-note/test.csv')

train = data_change(train)
test = data_change(test)

gamma = 0.05
a = 2

# nodes can be verified based on Set [5, 10, 25, 50, 100]
nodes = 100

Net = Neural_Net(input_shape=5, output_shape=1, hidden_layer=nodes)

"""Test Case"""
# Net.train(input=[1, 1, 1, 1], iter=1, gamma=gamma, a=a)

"""General Case"""
num_update = 1
Train_loss = []
epoch = 1
for i in range(epoch):
    train_set = train.copy()
    random.shuffle(train_set)
    for data in train_set:
        train_loss = Net.train(input=data, iter=num_update, gamma=gamma, a=a)
        Train_loss.append(train_loss)
        num_update += 1

    Train_error = testing(Net=Net, data=train)
    Test_error = testing(Net=Net, data=test)
    print(i, 'Training Error: ', Train_error)
    print(i, 'Testing Error: ', Test_error, '\n')

plt.plot(range(len(Train_loss)), Train_loss)
plt.show()

