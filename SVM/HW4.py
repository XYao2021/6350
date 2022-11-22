import matplotlib.pyplot as plt

from functions import *
import numpy as np
from scipy.optimize import minimize
import warnings


warnings.filterwarnings("ignore", category=RuntimeWarning)
train = open_file('bank-note/train.csv')
test = open_file('bank-note/test.csv')

train = data_change(train)
test = data_change(test)
#
# Modified parameters
a = 1
Set_gamma = [0.1, 0.5, 1, 5, 100]
signal = 2

# Hyper-parameters
Set_C = [100/873, 500/873, 700/873]
EPOCH = 100
# SVM Primal Form
# for C in Set_C:
#     train_Loss, test_Loss, w = SVM_primal(train, test, EPOCH, C, 2, a, signal)  # Primal SVM
#     print(C, 'Epoch Weights: ', w)
#     print(C, 'Training Loss: ', round(100*train_Loss, 3))
#     print(C, 'Testing Loss: ', round(100*test_Loss, 3), '\n')

# SVM Dual Form
# for C in Set_C:
#     w, train_acc, test_acc = SVM_dual(C, train, test)
#     print(C, 'w: ', w)
#     print(C, 'train acc: ', round(100*train_acc, 3))
#     print(C, 'test acc: ', round(100*test_acc, 3), '\n')



