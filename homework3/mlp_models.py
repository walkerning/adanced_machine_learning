import tensorflow as tf
from tf_iid_regression import nn_layer as _nn_layer

def simple_mlp_40_20_10_model(X, **kwargs):
    input_dim = kwargs.get("input_dim", 5)
    keep_prob = kwargs.get("keep_prob")
    with tf.name_scope("simple_mlp_model"):
        # 40-20-10
        hidden1, w1 = _nn_layer(X, input_dim, 40, "layer1")
        dropped = tf.nn.dropout(hidden1, keep_prob)
        hidden2, w2 = _nn_layer(dropped, 40, 20, "layer2")
        dropped2 = tf.nn.dropout(hidden2, keep_prob)
        hidden3, w3  = _nn_layer(dropped2, 20, 10, "layer3")
        dropped3 = tf.nn.dropout(hidden3, keep_prob)
        predict_y, w4 =  _nn_layer(dropped3, 10, 1, "layer4", act=tf.identity)
        return predict_y, w1 + w2 + w3 + w4

def simple_mlp_80_40_20_model(X, **kwargs):
    input_dim = kwargs.get("input_dim", 5)
    keep_prob = kwargs.get("keep_prob")
    with tf.name_scope("simple_mlp_model"):
        hidden1, w1 = _nn_layer(X, input_dim, 80, "layer1")
        dropped = tf.nn.dropout(hidden1, keep_prob)
        hidden2, w2 = _nn_layer(dropped, 80, 40, "layer2")
        dropped2 = tf.nn.dropout(hidden2, keep_prob)
        hidden3, w3  = _nn_layer(dropped2, 40, 20, "layer3")
        dropped3 = tf.nn.dropout(hidden3, keep_prob)
        predict_y, w4 =  _nn_layer(dropped3, 20, 1, "layer4", act=tf.identity)
        return predict_y, w1 + w2 + w3 + w4
