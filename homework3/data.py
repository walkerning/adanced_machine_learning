# -*- coding: utf-8 -*-
from __future__ import print_function
from functools import wraps
import itertools
from itertools import chain
import cPickle
import os

_cache_res_dict = {}
def cache(name):
    def _cache(func):
        @wraps(func)
        def _func(*args, **kwargs):
            arg_dict = dict(zip(func.func_code.co_varnames, args))
            arg_dict.update(kwargs)
            cache_id = name.format(**arg_dict)
            res = _cache_res_dict.get(cache_id, None)
            if res is None:
                _cache_res_dict[cache_id] = res = func(*args, **kwargs)
            return res
        return _func
    return _cache
        

class Paper(object):
    num_papers = 0
    papers_index_obj_map = {}
    #keys_index_map = {}

    _ABSTRACT = 1
    _CITES = 2
    _CONFERENCE = 4
    _YEAR = 8
    _AUTHOR_IDS = 16
    _TITLE = 32
    _INDEX = 64

    _WHOLE_FLAG = _ABSTRACT | _CITES | _CONFERENCE | _YEAR | _AUTHOR_IDS | _TITLE | _INDEX
    _COMMON_FLAG = _YEAR | _AUTHOR_IDS | _TITLE | _INDEX

    def __init__(self, index, title, author_ids, year, conference=None, cites=None, abstract=None):
        self.index = index
        self.title = title
        self.author_ids = author_ids
        self.year = year
        self.conference = conference
        self.cites = cites
        self.abstract = abstract
        self.cited_by = []

    @classmethod
    def get_paper_from_index(cls, index):
        return cls.papers_index_obj_map.get(index, None)

    @classmethod
    @cache("Papers.get_indexes_attr_dict_{name}")
    def get_indexes_attr_dict(cls, name):
        return {index:getattr(obj, name) for index, obj in cls.papers_index_obj_map.iteritems() if getattr(obj, name) is not None}

    @classmethod
    def get_indexes_by_attr(cls, name, val):
        return itertools.ifilter(lambda obj: getattr(obj, name) == val, cls.papers_index_obj_map.itervalues())
                
    # @classmethod
    # def get_indexes_with_infos(cls, infos):
    #     expected_info_flag = cls._flag(infos)
    #     if (cls._COMMON_FLAG^cls._WHOLE_FLAG) & expected_info_flag == 0:
    #         return cls.papers_index_obj_map.keys()
    #     flags = cls.keys_index_map.keys()
    #     return chain.from_iterable([cls.keys_index_map[key] for key in filter(lambda f: (f^cls._WHOLE_FLAG) & expected_info_flag == 0, flags)])

    @classmethod
    def _flag(cls, varnames):
        return reduce(lambda x, y: x | y, [getattr(cls, "_" + var_name.upper()) for var_name in varnames], 0)

    @classmethod
    def read_papers_from_file(cls, fname):
        print("Start reading papers from {}".format(fname))
        with open(fname, "r") as f:
            info_dict = {}
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    # another paper
                    cls.num_papers += 1
                    index = info_dict["index"]
                    assert index is not None
                    # FIXME: not sure use object or just plain dict?
                    # FIXME: problem of replicated index is disgarded for now
                    cls.papers_index_obj_map[index]= cls(**info_dict)
                    flag = cls._flag(info_dict.keys())
                    #cls.keys_index_map.setdefault(tuple((info_dict.keys())), []).append(index)
                    #cls.keys_index_map.setdefault(flag, []).append(index)
                    info_dict = {}
                if line.startswith("#*"):
                    info_dict["title"] = line[2:]
                if line.startswith("#@"):
                    info_dict["author_ids"] = Author.get_index_from_names(line[2:].split(", "))
                if line.startswith("#t"):
                    info_dict["year"] = int(line[2:])
                if line.startswith("#c"):
                    info_dict["conference"] = line[2:]
                if line.startswith("#index"):
                    info_dict["index"] = line[6:]
                if line.startswith("#!"):
                    info_dict["abstract"] = line[2:]
                if line.startswith("#%"):
                    info_dict.setdefault("cites", []).append(line[2:])
        print("Finish reading papers from {}".format(fname))

    @classmethod
    def pickle_to_file(cls, fname):
        with open(fname, "w") as f:
            cPickle.dump(cls.papers_index_obj_map, f)

    @classmethod
    def pickle_from_file(cls, fname):
        with open(fname, "r") as f:
            cls.papers_index_obj_map = cPickle.load(f)
            #cls.num_papers = len(cls.papers_index_obj_map)

class Author(object):
    authors_name_index_map = {}
    authors_index_obj_map = {}
    train_indexes = []
    test_indexes = []

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.target_cites_count = None

    @classmethod
    def init(cls, data_dir=".", author_fname="author.txt", citation_train_fname="citation_train.txt", 
             citation_test_fname="citation_test.txt"):
        cls.read_authors_from_file(os.path.join(data_dir, author_fname))
        cls.read_citation_train(os.path.join(data_dir, citation_train_fname))
        cls.read_citation_test(os.path.join(data_dir, citation_test_fname))

    @classmethod
    def read_authors_from_file(cls, fname):
        print("Start reading authors from {}".format(fname))
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                index, name = line.split("\t")
                author = cls(index, name)
                # Already verified, there are no author with the same name or index
                cls.authors_name_index_map[name] = index
                cls.authors_index_obj_map[index] = author
        print("Finish reading authors from {}".format(fname))

    @classmethod
    def read_citation_train(cls, fname):
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                index, name, cites_count = line.split("\t")
                cls.train_indexes.append(index)
                assert name == cls.authors_index_obj_map[index].name
                cls.authors_index_obj_map[index].target_cites_count = int(cites_count)

    @classmethod
    def read_citation_test(cls, fname):
        with open(fname, "r") as f:
            for line in f:
                line = line.strip()
                index, name = line.split("\t")
                assert name == cls.authors_index_obj_map[index].name
                cls.test_indexes.append(index)

    @classmethod
    def get_index_from_names(cls, names):
        if not isinstance(names, list):
            return cls.authors_name_index_map[names]
        return [cls.authors_name_index_map[name] for name in names]

    @classmethod
    def get_train_citations(cls):
        return [(index, cls.authors_index_obj_map[index].cites_count) for index in cls.train_indexes]

    @classmethod
    def get_author_from_index(cls, index):
        return cls.authors_index_obj_map.get(index, None)

def init(data_dir=".", paper_fname="paper.txt", author_fname="author.txt", citation_train_fname="citation_train.txt", 
         citation_test_fname="citation_test.txt"):
    Author.init(data_dir, author_fname, citation_train_fname, citation_test_fname)
    Paper.read_papers_from_file(os.path.join(data_dir, paper_fname))
    

def read_res_file(fname):
    res = {}
    with open(fname, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            index, number = line.split("\t")[:2]
            res[index] = float(number)
    return res
