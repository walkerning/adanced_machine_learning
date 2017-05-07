# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import cPickle
import tensorflow as tf
import numpy as np
import sys
import os
#from data import Author, Paper, init

FLAGS = None

class LearningRateComputer(object):
    def __init__(self, base_learning_rate, decay_rate, threshold=0.001, run_threshold=10):
        self.decay_rate = decay_rate
        self.learning_rate = base_learning_rate
        self.training_loss = []
        self.run = 0
        self.threshold = threshold
        self.run_threshold = run_threshold

    def compute_learning_rate(self, new_loss):
        self.training_loss.append(new_loss)
        if len(self.training_loss) == 1:
            return self.learning_rate
        if (self.training_loss[-2] - new_loss) / self.training_loss[-2] < self.threshold:
            self.run += 1
        if self.run >= self.run_threshold:
            self.learning_rate = self.learning_rate * self.decay_rate
            #print("learning rate: decay to {}. loss: {}".format(self.learning_rate, self.training_loss))
            self.run = 0
        return self.learning_rate
        

def _get_model(name):
    return getattr(sys.modules["__main__"], name + "_model")

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        #with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.
    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
      # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

def simple_mlp_model(X, **kwargs):
    """2 layer fc model with dropout

    2-GD: iteration 400: train loss: 1411316.75
    keep_prob = 0.9. val loss: 1104905.875

    2-GD: iteration 400: train loss: 1379497.0
    val loss: 1107279.75

    2-GD: iteration 400: train loss: 1413127.5
    val loss: 1100884.375
    不好训啊... 

    不带dropout = 1.  2-GD:
    iteration 40: train loss: 1440506.375
    iteration 50: train loss: 1297152.5

    iteration 400: train loss: 1254569.375
    val loss: 1034380.3125

    iteration 400: train loss: 1319160.5
         val loss: 1094623.75
    Saved model to ./models/savemodel1-3.model.

    3层就训不动了...


    v2 2-GD: 181-20-1; base lr 0.00001; run_threshold: 10; iteration 400. train loss: 730707.5625
    keep_prob=0.9. val loss: 668478.875.
    
    v2 3-GD: 181-20-10-1; base lr 0.00001; run_threshold: 10; iteration 400. keep_prob=0.9
    train loss:  749200.0625 val loss:  666006.4375
    
    v2 3-GD: 181-40-20-1; base lr 0.00001; run_threshold: 10; iteration 400. keep_prob=0.9
    train loss: 735889.3125 val loss: 665267.0625

    v3 3-GD: 186-40-20-1; base lr 0.00001; run_threshold: 10; iteration 400. keep_prob=0.9
    train loss: 649740.5625, val loss: 590059.0

    v3 3-GD: 186-80-40-20-1; base lr 0.00001; run_threshold: 10; iteration 400. keep_prob=0.9

    
    v3 3-GD: 186-40-40-20-5-average
    """
    # 最后改成一个average pooling嘛... 或者还是做成一个boost? 这两者等价吗
    # TODO: 加入其中几层initial from
    # 要不把这个模型单独提出来写吧...
    input_dim = kwargs.get("input_dim", 5)
    keep_prob = kwargs.get("keep_prob")
    with tf.name_scope("simple_mlp_model"):
        # hidden1 = nn_layer(X, input_dim, 10, "layer1")
        # dropped = tf.nn.dropout(hidden1, keep_prob)
        # predict_y = nn_layer(dropped, 10, 1, "layer2", act=tf.identity)

        # hidden1 = nn_layer(X, input_dim, 20, "layer1")
        # dropped = tf.nn.dropout(hidden1, keep_prob)
        # predict_y = nn_layer(dropped, 20, 1, "layer2", act=tf.identity)

        # 40-20-1
        hidden1 = nn_layer(X, input_dim, 40, "layer1")
        dropped = tf.nn.dropout(hidden1, keep_prob)
        hidden2 = nn_layer(dropped, 40, 20, "layer2")
        dropped2 = tf.nn.dropout(hidden2, keep_prob)
        # hidden3 = nn_layer(dropped2, 20, 20, "layer2-2")
        # dropped3 = tf.nn.dropout(hidden3, keep_prob)
        predict_y = nn_layer(dropped2, 20, 1, "layer3", act=tf.identity)
        # hidden3 = nn_layer(dropped2, 20, 10, "layer3")
        # predict_y = tf.reduce_mean(hidden3)

        # hidden1 = nn_layer(X, input_dim, 80, "layer1")
        # dropped = tf.nn.dropout(hidden1, keep_prob)
        # hidden2 = nn_layer(dropped, 80, 40, "layer2")
        # dropped2 = tf.nn.dropout(hidden2, keep_prob)
        # hidden3 = nn_layer(dropped2, 40, 20, "layer3")
        # dropped3 = tf.nn.dropout(hidden3, keep_prob)
        # predict_y = nn_layer(dropped3, 20, 1, "layer4", act=tf.identity)

        # dropped1 = tf.nn.dropout(hidden1, keep_prob)
        # hidden2 = nn_layer(dropped1, 10, 10, "layer2")
        # dropped2 = tf.nn.dropout(hidden2, keep_prob)
        # predict_y = nn_layer(dropped2, 10, 1, "layer3", act=tf.identity)
        return predict_y

def linear_regression_model(X, **kwargs):
    """
    sklearn.LinearRegression.fit: 1322561.8865403407
    here 200 iter: 1331947.75, lr: 0.0001
    iteration 400: train loss: 1309512.875
    val loss: 1088162.875
    模型太弱... 完全没有overfit
    """
    input_dim = kwargs.get("input_dim", 5)
    with tf.name_scope("linear_regression_model"):
        W = tf.Variable(tf.random_normal([input_dim, 1]), name="weight")
        b = tf.Variable(tf.random_normal([1]), name="bias")
        predict_y = tf.add(tf.matmul(X, W), b)
        return predict_y

def train():
    print("start loading features")
    train_features, train_targets, train_indexes, val_features, val_targets, val_indexes = cPickle.load(open(FLAGS.data_file, "r"))
    print("finish loading features")
    # normalize to var=1, mean=0
    if FLAGS.normalize_features:
        all_features = np.vstack((train_features, val_features))
        all_features_mean = np.mean(all_features, axis=0)
        all_features_std = np.std(all_features, axis=0)
        train_features = (train_features - all_features_mean) / all_features_std
        val_features = (val_features - all_features_mean) / all_features_std
    train_targets = train_targets[:, np.newaxis]
    val_targets = val_targets[:, np.newaxis]
    input_dim = train_features.shape[1]
    print("input dim: {}".format(input_dim))
    X = tf.placeholder("float", [None, input_dim])
    Y = tf.placeholder("float", [None, 1])
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
    predict_y = FLAGS.model(X, keep_prob=keep_prob, input_dim=input_dim)

    loss = tf.reduce_mean(tf.square(predict_y - Y)) # L2 loss
    tf.summary.scalar('loss', loss)

    saver = tf.train.Saver()
    # the path that the model weights will be loaded from or saved to
    save_model = os.path.join(FLAGS.save_model_dir, FLAGS.save_model_file)
    if FLAGS.eval:
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, save_model)
        train_res_fname = "train_res_" + FLAGS.run_var + ".txt"
        val_res_fname = "val_res_" + FLAGS.run_var + ".txt"
        
        train_loss_value, train_predict = sess.run([loss, predict_y], feed_dict={X: train_features, Y: train_targets, keep_prob: 1})
        
        val_loss_value, val_predict = sess.run([loss, predict_y], feed_dict={X: val_features, Y: val_targets, keep_prob: 1})
        print("train loss: {}, sqrt: {}; val loss: {}, sqrt: {}".format(train_loss_value, np.sqrt(train_loss_value), val_loss_value, np.sqrt(val_loss_value)))

        print("train_predict will be written to {}".format(train_res_fname))
        with open(train_res_fname, "w") as f:
            for ind, predict, target in zip(train_indexes, train_predict, train_targets):
                f.write("{}\t{}\t{}\n".format(ind, predict[0], target[0]))
        print("val will be written to {}".format(val_res_fname))
        with open(val_res_fname, "w") as f:
            for ind, predict, target in zip(val_indexes, val_predict, val_targets):
                f.write("{}\t{}\t{}\n".format(ind, predict[0], target[0]))

        if FLAGS.test is not None:
            with open(FLAGS.test, "r") as f:
                test_features, test_indexes = cPickle.load(f)
            if FLAGS.normalize_features:
                test_features = (test_features - all_features_mean) / all_features_std
            test_predict = sess.run(predict_y, feed_dict={X: test_features, keep_prob: 1})
            test_res_fname = "test_res_" + FLAGS.run_var + ".txt"
            print("test_predict will be written to {}".format(test_res_fname))
            with open(test_res_fname, "w") as f:
                for ind, predict in zip(test_indexes, test_predict):
                    f.write("{}\t{}\n".format(ind, predict[0]))
        return sess

    max_iter, test_iter, display_iter = FLAGS.max_iter, FLAGS.test_iter, FLAGS.display_iter

    ## optimizer
    # 先试试exponential decay
    global_step = tf.Variable(0, trainable=False, name="global_step")
    # start_learning_rate = FLAGS.learning_rate
    # learning_rate = tf.train.exponential_decay(start_learning_rate, global_step,
    #                                            20, 0.8, staircase=True)
    # 根据training set/validation set上的loss是否下降调整learning rate
    lrc = LearningRateComputer(FLAGS.learning_rate, 0.5, run_threshold=FLAGS.run_threshold)
    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.name_scope('train'):
        print("use optimizer: {}".format(FLAGS.optimizer))
        if FLAGS.optimizer == "gd":
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif FLAGS.optimizer == "momentum":
            optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
        elif FLAGS.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate, FLAGS.momentum)
        elif FLAGS.optimizer == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
            #learning_rate)
        grads = optimizer.compute_gradients(loss)
        for grad, varname in grads:
            print("add summary supervision for grad of {}".format(varname.name))
            with tf.name_scope(varname.op.name):
                variable_summaries(tf.square(grad))
        #train_step = optimizer.minimize(loss, global_step=global_step)
        train_step = optimizer.apply_gradients(grads, global_step=global_step)

    merged = tf.summary.merge_all()

    log_dir = "{}/{}".format(FLAGS.log_dir, FLAGS.run_var)
    # TODO: sgd?
    # with tf.Session() as sess:
    sess = tf.Session()
    #train_writer = tf.summary.FileWriter("{}/{}".format(FLAGS.log_dir + "/train", run_var), sess.graph_def)
    train_writer = tf.summary.FileWriter(log_dir + "/train", sess.graph)
    #train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph)
    #test_writer = tf.summary.FileWriter("{}/{}".format(FLAGS.log_dir + "/test", run_var))
    test_writer = tf.summary.FileWriter(log_dir + "/test")
    sess.run(tf.global_variables_initializer())
    if FLAGS.init_from is not None:
        print("start training from saved model: {}".format(FLAGS.init_from))
        saver.restore(sess, FLAGS.init_from)
    print("initial: train loss: {}".format(sess.run(loss, feed_dict={X: train_features, Y: train_targets, keep_prob: 1})))
    print("initial: val loss: {}".format(sess.run(loss, feed_dict={X: val_features, Y: val_targets, keep_prob: 1})))
    lr_value = FLAGS.learning_rate
    for iter in range(1, max_iter+1):
        if FLAGS.add_val:
            true_train_features = np.vstack((train_features, val_features))
            true_train_targets  = np.vstack((train_targets, val_targets))
        else:
            true_train_features = train_features
            true_train_targets = train_targets
        summary, loss_value, _ = sess.run([merged, loss, train_step], feed_dict={X: true_train_features, Y: true_train_targets, keep_prob: FLAGS.dropout, learning_rate: lr_value})
        lr_value = lrc.compute_learning_rate(loss_value)
        train_writer.add_summary(summary, iter)
        if iter % display_iter == 0:
            print("iteration {}: train loss: {}".format(iter, loss_value))
        if iter % test_iter == 0:
            summary, loss_value = sess.run([merged, loss], feed_dict={X: val_features, Y: val_targets, keep_prob: 1})
            test_writer.add_summary(summary, iter)
            print("\t val loss: {}".format(loss_value))
    print("last lr value: {}".format(lr_value))
    saver.save(sess, save_model)
    print("Saved model to {}.".format(save_model))
    #final_val_cost = sess.run(loss, feed_dict={X: val_features, Y: val_targets, keep_prob: 1})

    train_writer.close()
    test_writer.close()
    return sess

def main(_):
    tf.gfile.MakeDirs(FLAGS.log_dir)
    return train()

if __name__ == "__main__":
    import types
    attr_names = sys.modules["__main__"].__dict__.keys()
    available_models = [name[:-6] for name in attr_names if name.endswith("_model") and not name.startswith("_") and type(getattr(sys.modules["__main__"], name)) is types.FunctionType]

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", type=int, default=400,
                        help="Number of steps to run trainer.")
    parser.add_argument("--display_iter", type=int, default=10,
                        help="Number of steps to display training loss.")
    parser.add_argument("--test_iter", type=int, default=50,

                        help="Number of steps to test validation loss.")
    # TODO: learning rate can be ajusted
    parser.add_argument("--learning_rate", type=float, default=0.00001,
                        help="Initial learning rate")
    parser.add_argument("--dropout", type=float, default=0.9,
                        help="Keep probability for training dropout.")
    parser.add_argument("--run_var", default="default_running",
                        help="name to identify this running")

    parser.add_argument("--add-val", action="store_true", default=False,
                        help="use validation to train also")
    parser.add_argument("--init-from", default=None,
                        help="init training from saved model")
    parser.add_argument("--eval", default=False,
                        action="store_true",
                        help="eval model, not training. will ignore all other options except `model`, `save_model_dir/file`, `data_file`")
    parser.add_argument("--test", default=None,
                        help="test feature/index pkl file, must use with `--eval`")
    parser.add_argument("--model", choices=available_models, default="simple_mlp")
    parser.add_argument("--save_model_dir", default="./models/",
                        help="the dir to save the model")
    parser.add_argument("--save_model_file", default="model.model",
                        help="the file name to save the model")
    parser.add_argument("--optimizer", default="gd",
                        help="the optimizer type, gd / momentum / adam / rmsprop")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="the momentum of the optimizer with momentum (adam/momentum)")
    parser.add_argument("--normalize_features", action="store_true", default=False,
                        help="normalize features to var=1, mean=0")
    parser.add_argument(
        "--data_file",
        type=str,
        default="train_feature_v1.pkl",
        help="train/val data pickle file")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/tmp/tensorflow/papers/logs",
        help="Summaries log directory")
    # optimizer
    parser.add_argument(
        "--run_threshold",
        type=int,
        default=10,
        help="for learning rate ajustment")
    FLAGS, unparsed = parser.parse_known_args()
    print("use model: {}".format(FLAGS.model))
    FLAGS.model = _get_model(FLAGS.model)
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    sess = main([sys.argv[0]])
