# -*- coding: utf-8 -*-
# TODO: more averaging?
from __future__ import print_function
from data import read_res_file
import sys
import numpy as np
file_res1 = read_res_file(sys.argv[1])
file_res2 = read_res_file(sys.argv[2])
assert not set(file_res1.keys()).difference(set(file_res2.keys()))

with open(sys.argv[3], "w") as f:
    for index, num in file_res1.iteritems():
        f.write("{}\t{}\n".format(index, float(file_res2[index] + num)/2))
