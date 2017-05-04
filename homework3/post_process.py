# -*- coding: utf-8 -*-

# 一个简单的独立出来处理test得到的数据的脚本
import sys

# sys.argv[1]: 输入
# sys.argv[2]: 输出
less_than_0 = 0
with open(sys.argv[1], "r") as f:
    with open(sys.argv[2], "w") as wf:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ind, predict = line.split("\t")[:2]
            predict = int(float(predict))
            if predict < 0:
                less_than_0 += 1
                predict = 0
            wf.write("{}\t{}\n".format(ind, predict))

