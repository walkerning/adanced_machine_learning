# -*- coding: utf-8 -*-
import re
import sys
from matplotlib import pyplot as plt

plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
ax1.set_title("error rate")
ax2 = plt.subplot(2, 1, 2)
ax2.set_title("loss")
for fname in sys.argv[1:]:
    name = fname[:fname.index(".")]
    x = re.compile("loss = (?P<loss>[0-9.]+) .* metric = (?P<acc>[0-9.]+)% .*")
    acc_list = []
    loss_list = []
    with open(fname, "r") as f:
        for line in f:
            match = x.search(line)
            if match is not None:
                loss_list.append(match.group("loss"))
                acc_list.append(match.group("acc"))
    ax1.plot(range(1, len(acc_list)+1), acc_list, label=name)
    ax2.plot(range(1, len(loss_list)+1), loss_list, label=name)
plt.legend()
plt.show()
