import random
import numpy as np
import sklearn
from functions import *
import operator

"Problem 2 for dataset 'car', depth up to 6"
train_dataset = open_file('car/train.csv')
test_dataset = open_file('car/test.csv')

"Problem 2 for dataset 'bank', depth up to 16"
# train_dataset = open_file('bank/train.csv')
# test_dataset = open_file('bank/test.csv')

tree_depth = 5
# features, counts = get_features_and_num_with_unknown_feature(train_dataset)

dic = training(train_dataset, tree_depth)

training_test = testing(train_dataset, dic)
print('[TESTING RESULT FOR TRAINING DATASET]: ', training_test)
testing_test = testing(test_dataset, dic)
print('[TESTING RESULT FOR TESTING DATASET]: ', testing_test)
