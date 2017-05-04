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

print("error: ", np.sqrt(mean_squared_error(targets, file_res_list)))
