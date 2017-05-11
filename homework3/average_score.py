# -*- coding: utf-8 -*-
# TODO: more averaging?
from __future__ import print_function
from data import read_res_file
import sys
import numpy as np
# if len(sys.argv) > 4:
#     bl = float(sys.argv[4])
#     assert bl >= 0 and bl <= 1
# else:
#     bl = 0.5
print("read from {}".format(sys.argv[1]))
file_res1 = read_res_file(sys.argv[1])
file_res_list = [file_res1]
for i in range(len(sys.argv) - 3):
    print("read from {}".format(sys.argv[2 + i]))
    file_res = read_res_file(sys.argv[2 + i])
    assert not set(file_res.keys()).difference(set(file_res1.keys()))
    file_res_list.append(file_res)
# file_res1 = read_res_file(sys.argv[1])
# file_res2 = read_res_file(sys.argv[2])
# assert not set(file_res1.keys()).difference(set(file_res2.keys()))

with open(sys.argv[-1], "w") as f:
    for index, num in file_res1.iteritems():
        #f.write("{}\t{}\n".format(index, float((1-bl) * file_res2[index] + bl * num)))
        f.write("{}\t{}\n".format(index, np.mean([f_res[index] for f_res in file_res_list])))
