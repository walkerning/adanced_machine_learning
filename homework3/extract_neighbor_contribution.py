# -*- coding: utf-8 -*-
from __future__ import print_function
import cPickle
import numpy as np
from data import Author, Paper, init, read_res_file

total_cts = True
cts_truncate_fname = "cts_truncate_v16.txt"
cts_truncate_f = open(cts_truncate_fname, "w")
def prepare_total_cts(train_indexes_set, val_indexes_set):
    for index, p in Paper.papers_index_obj_map.iteritems():
        train_as = [Author.get_author_from_index(a_id) for a_id in p.author_ids if a_id in train_indexes_set]
        if train_as:
            p.max_cites_count = np.min([a.target_cites_count for a in train_as])
        else:
            p.max_cites_count = None
    trainval_indexes_set = train_indexes_set.union(val_indexes_set)
    for a_id in trainval_indexes_set:
        author = Author.get_author_from_index(a_id)
        p_mc = [Paper.get_paper_from_index(p_ind).max_cites_count for p_ind in author.all_papers]
        if None in p_mc:
            author.total_cts = author.target_cites_count
        else:
            author.total_cts = min(np.sum(p_mc), author.target_cites_count)
            if author.total_cts < author.target_cites_count:
                print("truncated ind: {}; ori: {}; cut to {}".format(author.index, author.target_cites_count, author.total_cts), file=cts_truncate_f)

def prepare_papers():
    def set_paper(self, order, index, cited_count):
        if not hasattr(self, "papers"):
            self.papers = {
                0: {},
                1: {},
                -1: {}
            }
            
        self.papers[order][index] = cited_count
    def set_all_paper(self, index):
        if not hasattr(self, "paper_count"):
            self.paper_count = 0
            self.all_papers = []
        self.all_papers.append(index)
        self.paper_count += 1

    for index, obj in Paper.papers_index_obj_map.iteritems():
        if obj.cites is not None:
            for cite in obj.cites:
                p = Paper.get_paper_from_index(cite)
                if p is None:
                    continue
                p.cited_by.append(index)
    for index, obj in Paper.papers_index_obj_map.iteritems():
        # Already verified: all paper have author_ids
        # 1-st author
        # FIXME: 直接存下这个作者的所有相关paper好像还好些...
        set_paper(Author.get_author_from_index(obj.author_ids[0]), 0, index, len(obj.cited_by))
        # 2-nd author
        if len(obj.author_ids) > 1:
            set_paper(Author.get_author_from_index(obj.author_ids[1]), 1, index, len(obj.cited_by))
        if len(obj.author_ids) > 2:
            set_paper(Author.get_author_from_index(obj.author_ids[-1]), -1, index, len(obj.cited_by))
        for author_id in obj.author_ids:
            author = Author.get_author_from_index(author_id)
            set_all_paper(author, index)

val_res_file = "sklearn_lr/results/v11.txt.val.average_gbrt.post"
test_res_file = "sklearn_lr/results/v11.txt.test.average_gbrt.post"
valtest_init = False
def prepare_valtest_init(valtest_indexes):
    init_dct = read_res_file(val_res_file)
    init_dct.update(read_res_file(test_res_file))
    for ind in valtest_indexes:
        Author.get_author_from_index(ind).init_cites_count = init_dct[ind]

use_val_for_test_feature = True
def main():
    smooth = 1
    mome = 0.8
    max_iters = 10
    # cts_fname = "cts_valtest_init.pkl"
    # feat_fname = "cts_feat_valtest_init.pkl"
    cts_fname = "cts_cut_tonly_use_val3.pkl"
    feat_fname = "cts_feat_cut_tonly_use_val3.pkl"

    init("..")
    # get train/val/test indexes
    train_indexes = cPickle.load(open("train_indexes.pkl", "r"))
    train_indexes_set = set(train_indexes)
    val_indexes = cPickle.load(open("val_indexes.pkl", "r"))
    val_indexes_set = set(val_indexes)
    #trainval_indexes_set = train_indexes_set.union(val_indexes_set)
    #trainval_indexes_set = train_indexes_set
    test_indexes = cPickle.load(open("test_indexes.pkl", "r"))
    test_indexes_set = set(test_indexes)
    valtest_indexes = np.hstack((val_indexes, test_indexes))
    valtest_indexes_set = set(valtest_indexes)

    print("prepare papers")
    prepare_papers()
    if valtest_init:
        print("prepare valtest init")
        prepare_valtest_init(valtest_indexes)
    print("initial ratio")
    no_papers_num = 0
    for a_id in Author.authors_index_obj_map:
        obj = Author.get_author_from_index(a_id)
        if not hasattr(obj, "all_papers"):
            #print("wocao ", a_id)
            no_papers_num += 1
            obj.all_papers = []
            obj.paper_count = 0
            obj.all_papers_cts = {}
            continue
        all_papers_cts = [len(Paper.get_paper_from_index(p_ind).cited_by) + smooth for p_ind in obj.all_papers]
        all_papers_cts = np.array(all_papers_cts, dtype=float) / np.sum(all_papers_cts)
        obj.all_papers_cts = dict(zip(obj.all_papers, all_papers_cts))
    print("no_papers_num: ", no_papers_num)
    # 假设所有的val/test都有一个初始的值了称为author.init_cites_count. val和test的now_cites_count会变化(post process), 先假设总citation不变把比例算出来. ...然后怎么做?怎么改变val/test的postprocess
    # 先算一次. 固定citation把比例提出来再算一次
    #先做这个吧: 不变的话就做pre extraction
    if total_cts:
        print("prepare total cts")
        #prepare_total_cts(trainval_indexes_set)
        prepare_total_cts(train_indexes_set, val_indexes_set)
    for iter in range(max_iters):
        # async update, do not restrict order(可能每次order近似一样)
        print("Iter #{}".format(iter+1))
        for a_id, author in Author.authors_index_obj_map.iteritems():
            if not hasattr(author, "all_papers_cts"):
                print("fcK: ", a_id)
            if a_id in valtest_indexes_set:
                now_cites_count = getattr(author, "init_cites_count", 0)
            elif a_id in train_indexes_set:
                #now_cites_count = author.target_cites_count
                now_cites_count = author.total_cts
            else:
                continue
            papers_cts = []
            for p_id in author.all_papers:
                p = Paper.get_paper_from_index(p_id)
                if use_val_for_test_feature and a_id in test_indexes_set:
                    col_as = [Author.get_author_from_index(col_id) for col_id in p.author_ids if col_id != a_id and (col_id in train_indexes_set or col_id in val_indexes_set)]
                else:
                    col_as = [Author.get_author_from_index(col_id) for col_id in p.author_ids if col_id != a_id and col_id in train_indexes_set]
                try:
                    if col_as:
                        # TODO: 加一个上限, 每轮修正target_cites_count在这些文章里的比例. 如果有单人文章或者colabrator都是val/test就可以暂时不修正... 全部都是train的话要修正
                        #papers_cts.append(mome * author.all_papers_cts[p_id] * now_cites_count + (1 - mome) * np.mean([col_a.all_papers_cts[p_id] * col_a.target_cites_count for col_a in col_as]))
                        updated_num = mome * author.all_papers_cts[p_id] * now_cites_count + (1 - mome) * np.mean([col_a.all_papers_cts[p_id] * col_a.total_cts for col_a in col_as])
                    else:
                        updated_num = author.all_papers_cts[p_id] * now_cites_count
                    # 这个truncate对于信息不全的paper和人是不是可能不太公平... 反而会抬高...
                    if total_cts and p.max_cites_count is not None and p.max_cites_count < updated_num:
                        updated_num = p.max_cites_count
                    papers_cts.append(updated_num)
                except:
                    import pdb
                    pdb.set_trace()
            if papers_cts:
                s = np.sum(papers_cts)
                # if s == np.nan or s == 0:
                #     import pdb
                #     pdb.set_trace()
                if s:
                    author.all_papers_cts = dict(zip(author.all_papers, np.array(papers_cts)/s))
    print("preparing cts dict")
    cts_dict = {a_id: author.all_papers_cts for a_id, author in Author.authors_index_obj_map.iteritems()}
    print("writing cts to {}".format(cts_fname))
    cPickle.dump(cts_dict, open(cts_fname, "w"))
    
    print("preparing cts features dict")
    feat_dict = {}
    for a_id, author in Author.authors_index_obj_map.iteritems():
        s = 0
        for p_id in author.all_papers:
            p = Paper.get_paper_from_index(p_id)
            if use_val_for_test_feature and a_id in test_indexes_set:
                col_as = [Author.get_author_from_index(col_id) for col_id in p.author_ids if col_id != a_id and (col_id in train_indexes_set or col_id in val_indexes_set)]
            else:
                col_as = [Author.get_author_from_index(col_id) for col_id in p.author_ids if col_id != a_id and col_id in train_indexes_set]
            if col_as:
                #s += author.all_papers_cts[p_id] * np.mean([col_a.all_papers_cts[p_id] * col_a.target_cites_count for col_a in col_as])
                s += author.all_papers_cts[p_id] * np.mean([col_a.all_papers_cts[p_id] * col_a.total_cts for col_a in col_as])
        feat_dict[a_id] = s
    print("writing cts features to {}".format(feat_fname))
    cPickle.dump(feat_dict, open(feat_fname, "w"))

if __name__ == "__main__":
    main()
