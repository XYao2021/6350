import random
import numpy as np
from functions import *


"label in the last column of each dataset"
train_dataset = open_file('car/train.csv')
test_dataset = open_file('car/test.csv')

tree_depth = 6
features, counts = get_features_and_num(dataset=train_dataset)
labels = features[-1]
org_features = features.copy()
# print(labels)
train_dataset = [train_dataset]
counts = [counts]
features = [features]
# org_features = features.copy()
print('org_features: ', org_features)
# print('counts: ', counts[0][:-1])
# print(train_dataset[0][0])
TREE = []

for i in range(tree_depth):
    new_counts = []
    new_f = []
    new_d = []
    Tree = [[] for _ in range(len(train_dataset))]
    for j in range(len(train_dataset)):
        # print(i, j, len(train_dataset), len(train_dataset[j]), train_dataset[j])
        # print(i, j, 'feature: ', len(features[j]), len(features), features[j])
        # print(i, j, 'counts: ', len(counts[j]), len(counts), counts[j])
        # print(i, j, 'data_set: ', len(train_dataset), len(train_dataset[j]), train_dataset[j])
        # print(i, j, 'counts[j][-1]: ', counts[j][-1])
        # print(i, j, len([i for i in counts[j][-1] if i != 0]))
        # print(j, len(train_dataset))
        if len([i for i in counts[j][-1] if i != 0]) != 1:
            E = Entropy(counts[j][-1])
            # print(j, 'Entropy: ', E)
            Gain = information_gain(features[j], counts[j], train_dataset[j], labels, E, org_features)
            # print('\n', j, 'gain: ', Gain)
            # print(new_features[Gain.index(max(Gain))],  Gain.index(max(Gain)), new_features.index(new_features[Gain.index(max(Gain))]))
            new = get_new_data(train_dataset[j], features[j][Gain.index(max(Gain))], org_features.index(features[j][Gain.index(max(Gain))]))
            # new_dataset.append(new)
            # print(j, 'new data: ', len(new), new)
            new_features = features[j].copy()
            Tree[j] = features[j][Gain.index(max(Gain))]
            print(j, 'feature remove: ', features[j][Gain.index(max(Gain))])
            print(j, 'Tree for now: ', Tree, '\n')
            new_features.remove(features[j][Gain.index(max(Gain))])
            # print(j, 'new_features: ', new_features)
            n_counts = []
            for n in new:
                # print(len(n), n)
                b = []
                # b.append(get_num_labels(n, labels))
                for nf in new_features[:-1]:
                    # print(nf, org_features.index(nf))
                    b.append(get_num_feature(n, nf, org_features.index(nf)))
                b.append(get_num_feature(n, labels, -1))
                n_counts.append(b)
            print(j, 'n_counts: ', len(n_counts), n_counts, '\n')
            # new_labels = n_counts[-1]
            # print('new_labels: ', new_labels)
            new_d += new
            new_counts += n_counts
            # new_f.append(new_features)
            new_f += [new_features for _ in range(len(new))]
        else:
            # new_counts.append([])
            # new_f.append([])
            Tree[j] = labels[counts[j][-1].index(sum(counts[j][-1]))]
            print(j, '[END] Label for this branch is ', labels[counts[j][-1].index(sum(counts[j][-1]))], '\n')
    # print(len(new))
    train_dataset = new_d
    # counts = [i for i in new_counts if i != []]
    counts = new_counts
    features = new_f
    TREE.append(Tree)
    print(i, 'Tree for this round: ', Tree)
    print(i, len(Tree))
    # print('for', i+1, 'NEW features: ', len(features), features, len(features))
    # print('for', i+1, 'NEW data: ', len(train_dataset), train_dataset)
    # print('for', i+1, 'NEW counts: ', len(counts), counts, 'for next round')
    print('[END] level ', i, '\n')

for t in range(len(TREE)):
    print('Length for level ', t, len(TREE[t]))
    print('Tree for level ', t, TREE[t], '\n')

#     else:
#         new_counts = []
#         for n in new:
#             b = []
#             # b.append(get_num_labels(n, labels))
#             for nf in new_features:
#                 b.append(get_num_feature(n, nf, features.index(nf)))
#             # print(b)
#             new_counts.append(b)
#         print('new_counts: ', new_counts)
#         for j in range(len(new_counts)):
#             # print(j, new_counts[j])
#             # print(j, new_features)
#             Es = Entropy(new_counts[j][-1])
#             # print(Es, '\n')
#             if Es != 0:
#                 gain = information_gain(new_features, new_counts[j], new[j], labels, Es)
#                 print(j, gain, new_features[gain.index(max(gain))])
#             else:
#                 print('label end here')

# E = Entropy(counts[-1])
# # print(E)
# # print(features.index(labels), len(features))
# Gain = information_gain(features, counts, train_dataset, labels, E)
# new = get_new_data(train_dataset, features[Gain.index(max(Gain))], Gain.index(max(Gain)))
# Tree.append(features[Gain.index(max(Gain))])
# # print(len(new), len(new[0]), len(new[1]), len(new[2]))
# # print(new)
# print(Tree)
# # print(Gain, features[Gain.index(max(Gain))])
#
# new_features = features.copy()
# new_features.remove(features[Gain.index(max(Gain))])
# print(new_features)
# # print(features.index(labels))
# new_counts = []
# for n in new:
#     b = []
#     # b.append(get_num_labels(n, labels))
#     for nf in new_features:
#         b.append(get_num_feature(n, nf, features.index(nf)))
#     # print(b)
#     new_counts.append(b)
# print('new_counts: ', new_counts)
# new1 = []
# for j in range(len(new_counts)):
#     # print(j, new_counts[j])
#     # print(j, new_features)
#     Es = Entropy(new_counts[j][-1])
#     # print(Es, '\n')
#     if Es != 0:
#         gain = information_gain(new_features, new_counts[j], new[j], labels, Es)
#         # print(j, gain, new_features[gain.index(max(gain))])
#         Tree.append(new_features[gain.index(max(gain))])
#         new1.append(get_new_data(new[j], new_features[gain.index(max(gain))], features.index(new_features[gain.index(max(gain))])))
#     else:
#         print('branch', j, 'end here, generate label')
#         Tree.append('end')
#         new1.append([])
# print(Tree)
# print(len(new1), len(new1[0]), len(new1[0][0]), len(new1[0][1]), len(new1[0][2]))
# a = []
# for f in range(len(new_features[:-1])):
#     a.append(get_num_feature(new, new_features[f], features.index(new_features[f])))
# print(a)
# new_gain = information_gain(features, counts, train_dataset, labels, E)
