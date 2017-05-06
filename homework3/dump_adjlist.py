# -*- coding: utf-8 -*-
from __future__ import print_function
import cPickle
import os
import argparse
from data import Author, Paper, init

def prepare_collabrators():
    # 需不需要加入cites?
    def add_col(author, ids):
        if not hasattr(author, "collabrations"):
            author.collabrations = {}
        for col_id in ids:
            if col_id == author.index:
                continue
            if col_id not in author.collabrations:
                author.collabrations[col_id] = 0
            author.collabrations[col_id] += 1

    for index, obj in Paper.papers_index_obj_map.iteritems():
        p = Paper.get_paper_from_index(index)
        if p is None:
            continue
        for a_id in p.author_ids:
            author = Author.get_author_from_index(a_id)
            add_col(author, p.author_ids)


def dump_adjlist(ADJ_FILE, map_file, threshold):
    number_index_mapping = {}
    index_number_mapping = {}
    print("deepwalk feature: dumping adjlist")
    number = 1
    for ind, author in Author.authors_index_obj_map.iteritems():
        if not hasattr(author, "collabrations"):
            #continue
            author._deepwalk_adjs = []
        else:
            author._deepwalk_adjs = [k for k, v in author.collabrations.iteritems() if v >= threshold]
        number_index_mapping[number] = ind
        index_number_mapping[ind] = number
        number += 1
    print("start writing adj list to {}".format(ADJ_FILE))
    with open(ADJ_FILE, "w") as f:
        for ind, author in Author.authors_index_obj_map.iteritems():
            # if not hasattr(author, "collabrations"):
            #     continue
            adjs = [index_number_mapping[c_ind] for c_ind in author._deepwalk_adjs]
            f.write("{} {}\n".format(index_number_mapping[ind], " ".join([str(x) for x in adjs])))
    print("writing mapping (index_number_mapping, number_index_mapping) to {}".format(map_file))
    with open(map_file, "w") as f:
        cPickle.dump((index_number_mapping, number_index_mapping), f)

"""
def get_features():

    # call deepwalk
    print("deepwalk feature: calling deepwalk")
    subprocess.check_call("deepwalk --format adjlist --input {} --output {} --walk-length 3 --window-size 3 --workers {} --undirected --number-walks 10 --representation-size {}".format(ADJ_FILE, TMP_FILE, args.deepwalk_workers, args.deepwalk_representation_size))
    # FIXME: skipgram model的window_size代表什么?
    # 读入TMP_FILE
    print("deepwalk feature: read result from deepwalk output")
    global _deepwalk_features
    with open(TMP_FILE, "r") as f:
        _ = f.readline()
        for line in f:
            fs = line.strip().split(" ")
            _deepwalk_features[number_index_mapping[int(fs[0])]] = np.array(fs[1:]).astype(np.float)
    
    print("deepwalk feature: dump cache file for future use")
    with open(CACHE_FILE, "w") as f:
        cPickle.dump(_deepwalk_features, f)
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_file", help="the file name to save adjlist to")
    parser.add_argument("map_file", help="number to index mapping file")
    parser.add_argument("--threshold", default=3, type=int, help="threshold to have connection")
    args = parser.parse_args()
    print("init data")
    here = os.path.dirname(os.path.abspath(__file__))
    datapath = os.path.dirname(here)
    print("data path: {}".format(datapath))
    init(datapath)
    print("prepare collabrators")
    prepare_collabrators()
    print("write adjlist to {}".format(args.save_file))
    dump_adjlist(args.save_file, args.map_file, args.threshold)
    
