import numpy as np
import matplotlib.pyplot as plt
from Linear_functions import *


train_dataset = open_file('concrete/train.csv')
test_dataset = open_file('concrete/test.csv')

# train_data, train_label = data_split(train_dataset)
# train_data = np.array(train_data)  # For simplicity
# train_label = np.array(train_label)

# test_data, test_label = data_split(test_dataset)
# test_data = np.array(test_data)  # For simplicity
# test_label = np.array(test_label)
# print(len(train_data), train_data[0], len(train_label), train_label[0])
# print(train_dataset[0])
'Initialize parameters'
features = 7
learning_rate = 0.01
w = np.zeros(features)  # Initial weight vector
iter = 1000
bond = 0.0001

'BatchGD'
w0, Loss = batch_grad(bond, learning_rate, w, train_dataset, features)
test_loss0 = testing(w0, test_dataset)

plt.plot(range(len(Loss)), Loss)
plt.xlabel('iterations')
plt.ylabel('Training Loss Value')
plt.title('Batch GD learning rate = 0.01')
plt.legend()
# plt.savefig('BatchGD.pdf')
plt.show()

# print(w0)
# print(test_loss0)
'SGD'
# w = np.zeros(features)
# w1, Loss1 = SGD(bond, w, learning_rate, train_dataset)
# test_loss1 = testing(w1, test_dataset)
#
# plt.plot(range(len(Loss1)), Loss1)
# plt.xlabel('iterations')
# plt.ylabel('Training Loss Value')
# plt.title('SGD learning rate = 0.01')
# plt.savefig('SGD.pdf')
# plt.show()
#
# print(w1)
# print(test_loss1)
# # plt.plot(range(len(Loss)), Loss)
# # plt.show()

'Optimal'
Opt = Optimal(train_dataset)
print(Opt)

