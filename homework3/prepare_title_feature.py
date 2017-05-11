# -*- coding: utf-8 -*-
from __future__ import print_function
import cPickle
import numpy as np
from data import Author, Paper, init, read_res_file
from extract_feature import prepare_papers
import gensim

def get_feature_titles():
    ps = Paper.papers_index_obj_map.values()
    titles = [[w.lower() for w in p.title.split()] for p in ps]
    inds = [p.index for p in ps]
    print("init corpos")
    corpos = gensim.corpora.Dictionary(titles)
    print("prepare vector space features")
    occurs = [corpos.doc2bow(t) for t in titles]
    print("calculate tfidf")
    tfidf = gensim.models.tfidfmodel.TfidfModel(occurs)
    print("fit skipgram/cobw word2vec embedding")
    model = prepare_embedding(titles)
    key_set = set(model.vocab.keys())
    print("prepare features")
    feat_dct = {}
    for i, t in enumerate(titles):
        t = list(set(t).intersection(key_set))
        if len(t) == 0:
            feat_dct[inds[i]] = np.zeros(64)
            continue
        vec = corpos.doc2bow(t)
        weight = tfidf[vec]
        # tf-idf weighted embedding mean
        #feat_dct[inds[i]] = np.mean([model[w] * weight[corpos.token2id[w]] for j, w in enumerate(t)], axis=0)
        feat_dct[inds[i]] = np.mean([model[w] * weight[j][1] for j, w in enumerate(t)], axis=0)
    return feat_dct
    
def prepare_embedding(titles):
    model = gensim.models.word2vec.Word2Vec(sentences=titles, size=64, window=5, min_count=5, workers=4)
    print("saving word2vec model to embedding.model")
    model.save("embedding.model")
    return model

def get_features_author_fields():
    paper_feat_dct = get_feature_titles()
    # average
    author_feat_dct = {}
    not_all_papers_num = 0
    for a_id, author in Author.authors_index_obj_map.iteritems():
        if not hasattr(author, "all_papers") or len(author.all_papers) == 0:
            not_all_papers_num += 1
            author_feat_dct[a_id] = np.zeros(64)
        else:
            author_feat_dct[a_id] = np.mean([paper_feat_dct[p_ind] for p_ind in author.all_papers], axis=0)
    print("not all_papers_num: {}".format(not_all_papers_num))
    cPickle.dump(paper_feat_dct, open("title_feature.pkl", "w"))
    cPickle.dump(author_feat_dct, open("author_field_feature.pkl", "w"))

if __name__ == "__main__":
    init("..")
    prepare_papers()
    get_features_author_fields()
