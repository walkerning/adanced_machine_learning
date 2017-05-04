# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import cPickle
import numpy as np
from data import Author, Paper, init

args = None
# 把1, 2和最后一个作者提出来. 如果文章的最后一个作者也就是1,2作者就当作1,2作者处理
def prepare_papers():
    def set_paper(self, order, index, cited_count):
        if not hasattr(self, "papers"):
            self.papers = {
                0: {},
                1: {},
                -1: {}
            }
        self.papers[order][index] = cited_count

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
            if not hasattr(author, "paper_count"):
                author.paper_count = 0
            author.paper_count += 1    

def prepare_collabrators():
    # 需不需要加入cites?
    def add_col(author, ids, year):
        if not hasattr(author, "collabrations"):
            author.collabrations = {}
            author.collabrations_by_year = {}
        if year not in author.collabrations_by_year:
            author.collabrations_by_year[year] = {}
        for col_id in ids:
            if col_id == author.index:
                continue
            if col_id not in author.collabrations:
                author.collabrations[col_id] = 0
            if col_id not in author.collabrations_by_year[year]:
                author.collabrations_by_year[year][col_id] = 0
            author.collabrations[col_id] += 1
            author.collabrations_by_year[year][col_id] += 1

    for index, obj in Paper.papers_index_obj_map.iteritems():
        p = Paper.get_paper_from_index(index)
        if p is None:
            continue
        for a_id in p.author_ids:
            author = Author.get_author_from_index(a_id)
            add_col(author, p.author_ids, p.year)

def get_feature_num_collabrator_by_year(author):
    # FIXME: 只看一/二作?
    # already verified, the year range is: 1936 ~ 2011
    # 取从1967年开始的数据. 1967年之前的paper占比为0.002535306438376776
    base_year = args.base_year
    num_collabrators_years_list = [0 for _ in range(2011-base_year+1)]
    if not hasattr(author, "collabrations_by_year"):
        return np.array(num_collabrators_years_list)
    
    for year, cols in author.collabrations_by_year.iteritems():
        ind = year - base_year
        if ind < 0:
            continue
        num_collabrators_years_list[ind] = len(cols)

    return np.array(num_collabrators_years_list)

def get_feature_all_paper_count(author):
    if not hasattr(author, "paper_count"):
        return np.array([0])
    return np.array([author.paper_count])

def get_feature_citation_count_by_year(author):
    base_year = getattr(args, "base_year", 1967)
    if not hasattr(author, "papers"):
        return [0 for _ in range((2011-base_year+1) * 4)]
    # already verified, the year range is: 1936 ~ 2011
    # 取从1967年开始的数据. 1967年之前的paper占比为0.002535306438376776
    cites_1st_years_list = [0 for _ in range(2011-base_year+1)]
    cites_2st_years_list = [0 for _ in range(2011-base_year+1)]
    num_1st_years_list = [0 for _ in range(2011-base_year+1)]
    num_2st_years_list = [0 for _ in range(2011-base_year+1)]
    for index, cited_count in author.papers[0].iteritems():
        p = Paper.get_paper_from_index(index)
        ind = p.year - base_year
        if ind < 0:
            continue
        cites_1st_years_list[ind] += cited_count
        num_1st_years_list[ind] += 1

    for index, cited_count in author.papers[1].iteritems():
        p = Paper.get_paper_from_index(index)
        ind = p.year - base_year
        if ind < 0:
            continue
        cites_2st_years_list[ind] += cited_count
        num_2st_years_list[ind] += 1
    return np.concatenate((num_1st_years_list, cites_1st_years_list, num_2st_years_list, cites_2st_years_list))

feature_dict = {
    "all_paper_count": get_feature_all_paper_count,
    "citation_count_by_year": get_feature_citation_count_by_year,
    "num_collabrator_by_year": get_feature_num_collabrator_by_year
}

feature_prepare_dict = {
    "all_paper_count": [prepare_papers],
    "citation_count_by_year": [prepare_papers],
    "deepwalk": [prepare_collabrators],
    "num_collabrator_by_year": [prepare_collabrators]
}

_guard_get_feature_deepwalk = None
number_index_mapping = {}
index_number_mapping = {}
_deepwalk_features = {}
def get_feature_deepwalk(author):
    """
    dump graph structure, and call deepwalk, use skip gram to generate collabrative latent feature of authors
    """
    global _deepwalk_features, _guard_get_feature_deep_walk
    global number_index_mapping, index_number_mapping
    _threshold = 3 # 合作过`_threshold`次或以上认为有关联
    CACHE_FILE = "deepwalk_cache.pkl"
    ADJ_FILE = "_deepwalk_adj.txt"
    TMP_FILE = "_deepwalk_tmpout.txt"
    if _guard_get_feature_deep_walk is None:
        if os.path.exists(CACHE_FILE):
            # 已经制作好了cache
            with open(CACHE_FILE, "r") as f:
                _deepwalk_features = cPickle.load(f)
        else:        
            # dump the adj list and also the mapping
            
            print("deepwalk feature: dumping adjlist")
            number = 0
            for ind, author in Author.authors_index_obj_map.iteritems():
                if not hasattr(author, "collabrations"):
                    continue
                author._deepwalk_adjs = [k for k, v in author.collabrations.iteritems() if v >= _threshold]
                number_index_mapping[number] = ind
                index_number_mapping[ind] = number
                number += 1
            with open(ADJ_FILE, "w") as f:
                for ind, author in Author.authors_index_obj_map.iteritems():
                    if not hasattr(author, "collabrations"):
                        continue
                    adjs = [index_number_mapping[c_ind] for c_ind in author._deepwalk_adjs]
                    f.write("{} {}\n".format(index_number_mapping[ind], " ".join(adjs)))
            # call deepwalk
            print("deepwalk feature: calling deepwalk")
            subprocess.check_call("deepwalk --format adjlist --input {} --output {} --walk-length 3 --window-size 3 --workers {} --number-walks 10 --representation-size {}".format(ADJ_FILE, TMP_FILE, args.deepwalk_workers, args.deepwalk_representation_size))
            # FIXME: skipgram model的window_size代表什么?
            # 读入TMP_FILE
            print("deepwalk feature: read result from deepwalk output")
            with open(TMP_FILE, "r") as f:
                _ = f.readline()
                for line in f:
                    fs = line.strip().split(" ")
                    _deepwalk_features[number_index_mapping[int(fs[0])]] = np.array(fs[1:]).astype(np.float)
            
            print("deepwalk feature: dump cache file for future use")
            with open(CACHE_FILE, "w") as f:
                cPickle.dump(_deepwalk_features, f)
        _guard_get_feature_deep_walk = True
    return _deepwalk_features[author.index]

def all_features(index, features, test=False):
    author = Author.get_author_from_index(index)
    ret = np.empty(0)
    for feature in features:
        nf = feature_dict[feature](author)
        if nf is None:
            return None, None
        ret = np.hstack((ret, nf)) 
    if not test:
        target = author.target_cites_count
    else:
        target = None
    return ret, target

def main():
    import sys
    legal_feature_names = feature_dict.keys()
    parser = argparse.ArgumentParser()
    parser.add_argument("save", help="save feature to file")
    parser.add_argument("-f", "--features", action="append", required=True,
                        help="all the features will be concat in order",
                        choices=legal_feature_names)
    parser.add_argument("--base-year", default=1967, type=int,
                        help="the base year of by-year")
    parser.add_argument("--train", required=True,
                        help="The pkl file that store all the train indexes")
    parser.add_argument("--val", required=True,
                        help="The pkl file that store all the val indexes")
    parser.add_argument("--test", required=True,
                        help="The pkl file that store all the test indexes")
    #for deepwalk
    parser.add_argument("--deepwalk-workers", default=1,
                        help="Number of parallel workers running `deepwalk`")

    parser.add_argument("--deepwalk-representation-size", default=32,
                        help="social reprensentation dimension")

    global args
    args = parser.parse_args()
    with open(args.train, "r") as f:
        train_indexes = cPickle.load(f)
    with open(args.val, "r") as f:
        val_indexes = cPickle.load(f)
    with open(args.test, "r") as f:
        test_indexes = cPickle.load(f)
    # init the data
    print("init the data")
    init("..")
    prepare_func_list = []
    for feature in args.features:
        for prepare in feature_prepare_dict.get(feature, []):
            if prepare not in prepare_func_list:
                prepare_func_list.append(prepare)
    print("running the preparing functions")
    for prepare in prepare_func_list:
        prepare()
    # import pdb
    # pdb.set_trace()
    print("running the get feature functions")
    # FIXME: to slow... 可以多进程执行嘛= =
    # finally, prepare all the features
    train_features1, train_target1 = all_features(train_indexes[0], args.features)
    val_features1, val_target1 = all_features(val_indexes[0], args.features)
    test_features1, _ = all_features(test_indexes[0], args.features, test=True)

    train_features = np.zeros((len(train_indexes), train_features1.size))
    val_features = np.zeros((len(val_indexes), val_features1.size))
    test_features = np.zeros((len(test_indexes), test_features1.size))
    train_features[0, :] = train_features1
    val_features[0, :] = val_features1
    test_features[0, :] = test_features1

    train_targets = [train_target1]
    val_targets = [val_target1]
    true_train_indexes = [train_indexes[0]]
    true_val_indexes = [val_indexes[0]]
    num = 1
    for index in train_indexes[1:]:
        feature, target = all_features(index, args.features)
        if feature is None:
            continue
        true_train_indexes.append(index)
        #train_features = np.vstack((train_features, feature))
        train_features[num, :] = feature
        train_targets.append(target)
        num += 1
        #if num % 1000 == 0:
        #print("cal train features: No. {}".format(num))
    train_features = train_features[:num, :]
    print("train features: total number: {}".format(num))
    num = 1
    for index in val_indexes[1:]:
        feature, target = all_features(index, args.features)
        if feature is None:
            continue
        true_val_indexes.append(index)
        #val_features = np.vstack((val_features, feature))
        val_features[num, :] = feature
        val_targets.append(target)
        num += 1
        #if num % 1000 == 0:
        #    print("cal val features: No. {}".format(num))
    val_features = val_features[:num, :]
    print("val features: total number: {}".format(num))
    train_targets = np.array(train_targets)
    val_targets = np.array(val_targets)

    for num, index in enumerate(test_indexes[1:]):
        feature, _ = all_features(index, args.features, test=True)
        if feature is None:
            continue
        test_features[num+1, :] = feature

    # write into pkl file
    print("Writing features `(train_features, train_targets, train_indexes, val_features, val_targets, val_indexes)` to file {}.train".format(args.save))
    with open(args.save + ".train", "w") as f:
        cPickle.dump((train_features, train_targets, true_train_indexes, val_features, val_targets, true_val_indexes), f)
    print("Writing features `(test_features, test_indexes)` to file {}.test".format(args.save))
    with open(args.save + ".test", "w") as f:
        cPickle.dump((test_features, test_indexes), f)



"""
TODO:
preprocess中加入发布的期刊影响力的feature. 用每个期刊的总引用数衡量

"""
if __name__ == "__main__":
    main()
