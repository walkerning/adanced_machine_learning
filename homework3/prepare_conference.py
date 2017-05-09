# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import argparse
import cPickle
import numpy as np
from data import Author, Paper, init

def prepare_conferences(base_year):
    conference_dict = {}
    conference_score_dict = {}
    for index, obj in Paper.papers_index_obj_map.iteritems():
        if obj.cites is not None:
            for cite in obj.cites:
                p = Paper.get_paper_from_index(cite)
                if p is None:
                    continue
                p.cited_by.append(index)
    for index, obj in Paper.papers_index_obj_map.iteritems():
        if obj.conference is None:
            continue
        # FIXME: 有些会议后面接着不同的数字代表什么??? 需不需要先处理. 两个都试试
        # 看起来有一些很少文章的所谓conference诶... 感觉可能在这种噪音情况下基于boost ensemble的方法会比直接feature好
        # TODO: 把conference的平均每年文章数目也加入?
        if obj.year < base_year:
            continue
        conference_dict.setdefault(obj.conference, []).append(len(obj.cited_by))
    for conference, cited_list in conference_dict.iteritems():
        conference_score_dict[conference] = (np.mean(cited_list), np.median(cited_list))
    return conference_score_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("save")
    parser.add_argument("--base-year", default=1967, type=int)
    args = parser.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.dirname(here)
    print("data path: {}".format(datapath))
    init(datapath)
    print("prepare conferences")
    score_dict = prepare_conferences(args.base_year)
    print("dump score dict to {}".format(args.save))
    with open(args.save, "w") as f:
        cPickle.dump(score_dict, f)

if __name__ == "__main__":
    main()
