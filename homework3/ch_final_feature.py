# -*- coding: utf-8 -*-

from __future__ import print_function
import cPickle
import sys
assert len(sys.argv) == 4

print("loading new last feature from {}".format(sys.argv[2]))
feat = cPickle.load(open(sys.argv[2], "r"))
print("loading train feature from {}.train".format(sys.argv[1]))
train_features, train_targets, train_indexes, val_features, val_targets, val_indexes = cPickle.load(open(sys.argv[1] + ".train", "r"))
print("loading test feature from {}.test".format(sys.argv[1]))
test_features, test_indexes = cPickle.load(open(sys.argv[1] + ".test", "r"))

train_features[:, -1] = [feat[ind] for ind in train_indexes] 
val_features[:, -1] = [feat[ind] for ind in val_indexes] 
test_features[:, -1] = [feat[ind] for ind in test_indexes] 

print("Writing features `(train_features, train_targets, train_indexes, val_features, val_targets, val_indexes)` to file {}.train".format(sys.argv[3]))
with open(sys.argv[3] + ".train", "w") as f:
    cPickle.dump((train_features, train_targets, train_indexes, val_features, val_targets, val_indexes), f)

print("Writing features `(test_features, test_indexes)` to file {}.test".format(sys.argv[3]))
with open(sys.argv[3] + ".test", "w") as f:
    cPickle.dump((test_features, test_indexes), f)

