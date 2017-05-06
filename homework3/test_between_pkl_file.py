# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from sklearn.metrics import mean_squared_error
from data import read_res_file
import sys
import cPickle

# sys.argv[1]: pkl file. have targets
# sys.argv[2]: result file

with open(sys.argv[1], "r") as f:
    features, targets, indexes = cPickle.load(f)

file_res = read_res_file(sys.argv[2])
file_res_list = [file_res[index] for index in indexes]

diff = np.abs(targets - file_res_list)
orders = np.argsort(diff)[::-1]
print("error: ", np.sqrt(mean_squared_error(targets, file_res_list)))

print("最不准确的:")
for order_ind in range(10):
    order = orders[order_ind]
    print("No.{} {}: res {} target {}".format(order_ind, indexes[order], file_res_list[order], targets[order]))

