from functions import *
import random
import numpy as np

train = open_file('bank-note/train.csv')
test = open_file('bank-note/test.csv')

print(len(train), len(test))

EPOCH = 10  # Total Epochs
LR = 0.1  # Learning Rate
w = [0 for i in range(len(train[0][:-1]))]
b = 0

# standard_perceptron(train, test, EPOCH, LR, w, b)
vote_perceptron(train, test, EPOCH, LR, w, b)
# Average_perceptron(train, test, EPOCH, LR, w, b)

