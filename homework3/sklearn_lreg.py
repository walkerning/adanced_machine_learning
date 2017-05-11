# -*- coding: utf-8 -*-
from __future__ import print_function
import cPickle
import argparse
import argparse
import numpy as np
# a stupid env fix on my machine...
#import sys
#sys.path.insert(0, "/home/foxfi/anaconda2/envs/lasso/lib/python2.7/site-packages")
import sklearn
from sklearn import linear_model
print("sklearn: ", sklearn.__path__)
from sklearn.metrics import mean_squared_error

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--res-file", required=True)

    subparser = parser.add_subparsers(dest="phase")
    train_parser = subparser.add_parser("train")
    train_parser.add_argument("--model", default="lr", help="the training model", choices=["lr", "lasso"])
    train_parser.add_argument("--data-file", required=True)
    train_parser.add_argument("--lasso-alpha", default=0.01, type=float)
    test_parser = subparser.add_parser("test")
    test_parser.add_argument("--test-from", default=None)

    args = parser.parse_args()

    if args.phase == "test":
        assert args.test_from is not None
        with open(args.test_from, "r") as f:
            test_features, test_indexes = cPickle.load(f)
        with open(args.model_file, "r") as f:
            reg = cPickle.load(f)
        test_predict = reg.predict(test_features)
        test_res_fname = args.res_file + ".test"
        print("writing result to {}".format(test_res_fname))
        with open(test_res_fname, "w") as f:
            for ind, predict in zip(test_indexes, test_predict):
                f.write("{}\t{}\n".format(ind, predict))
        return
    print("Loading features")
    train_features, train_targets, train_indexes, val_features, val_targets, val_indexes = cPickle.load(open(args.data_file, "r"))
    if args.model == "lr":
        reg = linear_model.LinearRegression()
    elif args.model == "lasso":
        reg = linear_model.Lasso(alpha=args.lasso_alpha)
    print("Start fitting regression")
    reg.fit(train_features, train_targets)
    print("Finish fitting regression")
    print(reg.coef_)
    print(reg.intercept_)
    print("saving model to {}".format(args.model_file))
    with open(args.model_file, "w") as f:
        cPickle.dump(reg, f)

    train_predict = reg.predict(train_features)
    train_res_fname =  args.res_file + ".train"
    print("Writing train results to {}".format(train_res_fname))
    print("training error: ", mean_squared_error(train_targets, train_predict))
    with open(train_res_fname, "w") as f:
        for ind, predict, target in zip(train_indexes, train_predict, train_targets):
            f.write("{}\t{}\t{}\n".format(ind, predict, target))

    val_predict = reg.predict(val_features)
    val_res_fname = args.res_file + ".val"
    print("Writing val results to {}".format(val_res_fname))
    with open(val_res_fname, "w") as f:
        for ind, predict, target in zip(val_indexes, val_predict, val_targets):
            f.write("{}\t{}\t{}\n".format(ind, predict, target))
    print("val error: ", mean_squared_error(val_targets, val_predict))


if __name__ == "__main__":
    main()
