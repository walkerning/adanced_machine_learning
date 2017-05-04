#-*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
import time
import cPickle
import argparse
import yaml
import numpy as np
import subprocess
import multiprocessing
from multiprocessing import Pool
from exceptions import Exception
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

defaults = {
    "train": {
        "n_estimators": 100,
        "max_depth": 3,
        "lr": 0.1,
        "save_dir": "models/gbrt/",
        "res_dir": "results/gbrt/",
        "loss": "ls"
    },
    "test": {
        "res_dir": "results/gbrt/"
    }
}

types = {
    "n_estimators": int,
    "max_depth": int,
    "lr": float
}
class SimpleConfig(object):
    def __init__(self, dct, phase, index, log_dir):
        self.dct = dct
        self.phase = phase
        self.index = index
        self.log_dir = log_dir

    def __getattr__(self, attr_name):
        attr = self.dct.get(attr_name, defaults[self.phase].get(attr_name, None))
        if attr is None:
            raise Exception("`{}` not exists in config".format(attr_name))
        attr = types.get(attr_name, str)(attr)
        return attr
    
    def __repr__(self):
        if self.phase == "train":
            return "index: {}; num: {}; depth: {}; lr: {}; save_dir: {}; res_dir {}".format(self.index, self.n_estimators, self.max_depth, self.lr, self.save_dir, self.res_dir)
        return ""

def train(args, data_dict=None):
    run_var = int(time.time())
    if isinstance(args, SimpleConfig):
        pid = os.getpid()
        print("Process {}: run training: {}".format(pid, args), file=sys.stderr)
        # redirect output to log file
        #sys.stdout = open(os.path.join(args.log_dir, "{}.log".format(args.index)), "w")
        run_var = "{}_{}".format(args.index, run_var)
    print("loading features")
    if data_dict is not None:
        if args.data_file not in data_dict:
            data_dict[args.data_file] = cPickle.load(open(args.data_file, "r"))
        train_features, train_targets, train_indexes, val_features, val_targets, val_indexes = data_dict[args.data_file]
    else:
        train_features, train_targets, train_indexes, val_features, val_targets, val_indexes = cPickle.load(open(args.data_file, "r"))

    print("setup regressor")
    regressor = GradientBoostingRegressor(n_estimators=args.n_estimators, learning_rate=args.lr,
                                          max_depth=args.max_depth, random_state=0, loss=args.loss, warm_start=True)
    # FIXME: what is max_depth of decision tree.
    print("fitting regressor")
    regressor.fit(train_features, train_targets)
    model_path = os.path.join(args.save_dir, "n{}_d{}_lr{}-{}.model".format(args.n_estimators, args.max_depth, args.lr, run_var))
    print("saving regressor to {}".format(model_path))
    with open(model_path, "w") as f:
        #cPickle.dump(regressor.get_params(deep=True), f)
        cPickle.dump(regressor, f)

    print("training score: ", regressor.train_score_[-1])

    train_predict = regressor.predict(train_features)
    print("training error: ", mean_squared_error(train_targets, train_predict))
    train_res_fname = os.path.join(args.res_dir, "train_res_gbrt_n{}_d{}_lr{}-{}.txt".format(args.n_estimators, args.max_depth, args.lr, run_var))
    print("Writing train results to {}".format(train_res_fname))
    with open(train_res_fname, "w") as f:
        for ind, predict, target in zip(train_indexes, train_predict, train_targets):
            f.write("{}\t{}\t{}\n".format(ind, predict, target))
    
    val_predict = regressor.predict(val_features)
    print("val error: ", mean_squared_error(val_targets, val_predict))
    val_res_fname = os.path.join(args.res_dir, "val_res_gbrt_n{}_d{}_lr{}-{}.txt".format(args.n_estimators, args.max_depth, args.lr, run_var))
    print("Writing val results to {}".format(val_res_fname))
    with open(val_res_fname, "w") as f:
        for ind, predict, target in zip(val_indexes, val_predict, val_targets):
            f.write("{}\t{}\t{}\n".format(ind, predict, target))

def test(args):
    run_var = int(time.time())
    with open(args.load_from, "r") as f:
        regressor = cPickle.load(f)
    with open(args.test_from, "r") as f:
        test_features, test_indexes = cPickle.load(f)
        test_predict = regressor.predict(test_features)
        #test_res_fname = os.path.join(args.res_dir, "test_res_gbrt_n{}_d{}_lr{}-{}.txt".format(args.n_estimators, args.max_depth, args.lr, run_var))
        test_res_fname = args.res_file
        print("test_predict will be written to {}".format(test_res_fname))
        with open(test_res_fname, "w") as f:
            for ind, predict in zip(test_indexes, test_predict):
                f.write("{}\t{}\n".format(ind, predict))

def batch_train(arg, data_dict=None):
    config, index, log_dir = arg
    argconfig = SimpleConfig(config, "train", index, log_dir)
    train(argconfig, data_dict=data_dict)

def batch(num_processes, configs, log_dir):
    pass_args = zip(configs, range(len(configs)), [log_dir for _ in range(len(configs))])
    pool = Pool(num_processes)
    print("batch training mode: starting pool", file=sys.stderr)
    pool.map(batch_train, pass_args)

def batch_sequential(configs, log_dir):
    data_dict = {}
    for ind, config in enumerate(configs):
        batch_train((config, ind, log_dir), data_dict)

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="phase")
    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--load-from", required=True)
    test_parser.add_argument("--test-from", required=True)
    #test_parser.add_argument("--res-dir", default=defaults["test"]["res_dir"])
    test_parser.add_argument("--res-file", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-file", required=True)
    train_parser.add_argument("--n-estimators", default=defaults["train"]["n_estimators"], type=int)
    train_parser.add_argument("--max-depth", default=defaults["train"]["max_depth"], type=int)
    train_parser.add_argument("--lr", default=defaults["train"]["lr"], type=float)
    train_parser.add_argument("--save-dir", default=defaults["train"]["save_dir"])
    train_parser.add_argument("--res-dir", default=defaults["train"]["res_dir"])
    train_parser.add_argument("--loss", default=defaults["train"]["loss"])
    batch_parser = subparsers.add_parser("batch")
    batch_parser.add_argument("conf", help="yaml file of list of configuration")
    batch_parser.add_argument("--parallel", action="store_true")
    batch_parser.add_argument("--n-workers", default=None, help="number of workers, default: #cpu/4", type=int)
    batch_parser.add_argument("--log-dir", required=True)

    args = parser.parse_args()
    if args.phase == "batch":
        num_process = args.n_workers or multiprocessing.cpu_count() / 4
        with open(args.conf, "r") as f:
            configs = yaml.load(f)
            assert isinstance(configs, list)
        if not args.parallel:
            batch_sequential(configs, args.log_dir)
        else:
            batch(num_process, configs, args.log_dir)
    elif args.phase == "train":
        train(args)
    else:
        test(args)
    

if __name__ == "__main__":
    main()

"""
loading features
setup regressor
fitting regressor
training score:  541839.533454
training error:  541839.533454
Writing train results to results/gbrt/train_res_gbrt_n100_d3_lr0.1-1493817979.txt
val error:  628698.649239
Writing val results to results/gbrt/val_res_gbrt_n100_d3_lr0.1-1493817979.txt

loading features
setup regressor
fitting regressor
training score:  541839.533454
training error:  541839.533454
Writing train results to results/gbrt/train_res_gbrt_n100_d3_lr0.1-1493825840.txt
val error:  628698.649239
Writing val results to results/gbrt/val_res_gbrt_n100_d3_lr0.1-1493825840.txt
"""
