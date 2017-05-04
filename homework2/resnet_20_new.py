# -*- coding: utf-8 -*-
from __future__ import print_function
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from cntk.layers import default_options, Convolution, MaxPooling, AveragePooling, Dropout, BatchNormalization, Dense, Sequential, For
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
import cntk.io.transforms as xforms 
from cntk.initializer import glorot_uniform, he_normal
from cntk import Trainer
from cntk.learners import nesterov, momentum_sgd, adagrad, adam, learning_rate_schedule, UnitType, momentum_as_time_constant_schedule, rmsprop
from cntk import cross_entropy_with_softmax, classification_error, relu, input_variable, softmax, element_times
#from cntk.ops import cross_entropy_with_softmax, classification_error, relu, input_variable, softmax, element_times
from cntk.logging import *
#from cntk.utils import *

image_height = 32
image_width  = 32
num_channels = 3
num_classes  = 10

#
# Define the reader for both training and evaluation action.
#
def create_reader(map_file, mean_file, train):
    if not os.path.exists(map_file) or not os.path.exists(mean_file):
        raise RuntimeError("Files for training/testing not exists.")

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', side_ratio=0.8) # train uses data augmentation (translation only)
        ]
    transforms += [
        xforms.scale(width=image_width, height=image_height, channels=num_channels, interpolations='linear'),
        xforms.mean(mean_file)
    ]
    # deserializer
    return MinibatchSource(ImageDeserializer(map_file, StreamDefs(
        features = StreamDef(field='image', transforms=transforms), # first column in map file is referred to as 'image'
        labels   = StreamDef(field='label', shape=num_classes)      # and second as 'label'
    )))

#
# Train and evaluate the network.
#
def train_and_evaluate(reader_train, reader_test, max_epochs, model_func):
    start_time = time.time()
    print(start_time)
    # Input variables denoting the features and label data
    input_var = input_variable((num_channels, image_height, image_width))
    label_var = input_variable((num_classes))

    # Normalize the input
    feature_scale = 1.0 / 256.0
    input_var_norm = element_times(feature_scale, input_var)
    
    # apply model to input
    z = model_func(input_var_norm, out_dims=10)

    #
    # Training action
    #

    # loss and metric
    # surrogate loss objective
    ce = cross_entropy_with_softmax(z, label_var)
    # true objective
    pe = classification_error(z, label_var)

    # training config
    epoch_size     = 50000
    minibatch_size = 128

    # Set training parameters
    # maybe or decay the learning rate when loss stop to decrease for some epochs
    #lr_per_minibatch       = learning_rate_schedule([0.01]*20 + [0.003]*20 + [0.001]*20 + [0.0001] * 20 + [0.00001]*20 + [0.000001], UnitType.minibatch, epoch_size)
    # step epoch 20, base lr 0.01, step decay 0.1
    # summarize的方法...
    #lr_per_minibatch       = learning_rate_schedule(sum([[0.01 * lr] * 20 for lr in 0.1 ** np.arange(5)], []), UnitType.minibatch, epoch_size)
    lr_per_minibatch       = learning_rate_schedule([1.0]*80 + [0.1]*40 + [0.01], UnitType.minibatch, epoch_size)
    momentum_time_constant = momentum_as_time_constant_schedule(-minibatch_size/np.log(0.9))
    l2_reg_weight          = 0.0001
    
    # trainer object
    #learner = momentum_sgd(z.parameters, 
    #                        lr = lr_per_minibatch, momentum = momentum_time_constant, 
    #                        l2_regularization_weight=l2_reg_weight)
    #learner = adagrad(z.parameters, lr_per_minibatch, l2_regularization_weight=l2_reg_weight)
    learner = adam(z.parameters, lr_per_minibatch, momentum_time_constant, l2_regularization_weight=l2_reg_weight, adamax=True)
    #learner = nesterov(z.parameters, lr_per_minibatch, momentum_time_constant, l2_regularization_weight=l2_reg_weight)
    #learner = rmsprop(z.parameters, lr_per_minibatch, gamma=0.95, inc=1.5, dec=2, max=1, min=0.00001, l2_regularization_weight=l2_reg_weight)

    progress_printer = ProgressPrinter(tag='Training', num_epochs=max_epochs)

    # trainer这里为什么要第一个先输入z... 感觉这个语义是不是限制的有点死...
    # 看一看这边对data graph的eval是怎么resolve dependency. 然后trainer限制了哪些东西...
    # summary系统感觉有满多可以改...
    # tensorflow里train_step也是一个data graph中的节点. 所以也可以用sess.run(train_step)之类的来运行. 然后learning_rate也是一个Tensor(一个计算节点的输出)作为train_step的依赖. 所以可以很智能的计算learning rate.. 就有一堆可以根据global step或者训练error再计算learning rate的方便的Operation... 感觉这边就很局限. 因为trainer与数据流图分开了...trainer只是负责调用DAG数据流图上的backward, forward. 设计的时候并没有设计有一个动态learning rate的数据流图节点作为输入... 而是一个固定的schedule...不过这个可以做...是可以做的...确认一下Trainer里对learning rate schedule的访问...如果是按iterable方法访问的倒是可以继承patch...如果不是。就需要改Trainer里对leanring rate的获取, generalize成函数调用... (要传什么context呢)
    # 所以确实还是tensorflow的设计比较科学... learning rate和负责具体的parameter update strategy的trainer都是数据图中的节点. 并不是说trainer作为一个GOD来协调这样太不通用了. 要决定哪些东西可以动态改. 不同的training方法可能有不同的可以动态决定的参数. 都要这些子trainer 自己去adhoc的调用类似lr()的方法... 
    # 如果统一到数据流图里. 每次动态得到lr之类的参数并用于training, 就是data graph的dependency解析和inference机制. 不需要自己做多余的事情...
    
    trainer = Trainer(z, (ce, pe), [learner], [progress_printer])

    # define mapping from reader streams to network inputs
    input_map = {
        input_var: reader_train.streams.features,
        label_var: reader_train.streams.labels
    }

    log_number_of_parameters(z) ; print()

    # perform model training
    batch_index = 0
    plot_data = {'batchindex':[], 'loss':[], 'error':[]}
    for epoch in range(max_epochs):       # loop over epochs
        sample_count = 0
        while sample_count < epoch_size:  # loop over minibatches in the epoch
            data = reader_train.next_minibatch(min(minibatch_size, epoch_size - sample_count), input_map=input_map) # fetch minibatch.
            trainer.train_minibatch(data)                                   # update model with it

            sample_count += data[label_var].num_samples                     # count samples processed so far
            
            # For visualization...            
            # 这个summary感觉看起来不太智能...
            # 
            plot_data['batchindex'].append(batch_index)
            plot_data['loss'].append(trainer.previous_minibatch_loss_average)
            plot_data['error'].append(trainer.previous_minibatch_evaluation_average)
            
            batch_index += 1
        trainer.summarize_training_progress()
    end_time = time.time()
    print(end_time)
    print("took {} ms".format((end_time-start_time)*1000.0))
    #
    # Evaluation action
    #
    epoch_size     = 10000
    minibatch_size = 16

    # process minibatches and evaluate the model
    metric_numer    = 0
    metric_denom    = 0
    sample_count    = 0
    minibatch_index = 0

    while sample_count < epoch_size:
        current_minibatch = min(minibatch_size, epoch_size - sample_count)

        # Fetch next test min batch.
        data = reader_test.next_minibatch(current_minibatch, input_map=input_map)

        # minibatch data to be trained with
        metric_numer += trainer.test_minibatch(data) * current_minibatch
        metric_denom += current_minibatch

        # Keep track of the number of samples processed so far.
        sample_count += data[label_var].num_samples
        minibatch_index += 1

    print("")
    print("Final Results: Minibatch[1-{}]: errs = {:0.1f}% * {}".format(minibatch_index+1, (metric_numer*100.0)/metric_denom, metric_denom))
    print("")
    
    # # Visualize training result:
    # window_width            = 32
    # loss_cumsum             = np.cumsum(np.insert(plot_data['loss'], 0, 0)) 
    # error_cumsum            = np.cumsum(np.insert(plot_data['error'], 0, 0)) 

    # # Moving average.
    # plot_data['batchindex'] = np.insert(plot_data['batchindex'], 0, 0)[window_width:]
    # plot_data['avg_loss']   = (loss_cumsum[window_width:] - loss_cumsum[:-window_width]) / window_width
    # plot_data['avg_error']  = (error_cumsum[window_width:] - error_cumsum[:-window_width]) / window_width
    
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(plot_data["batchindex"], plot_data["avg_loss"], 'b--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Loss')
    # plt.title('Minibatch run vs. Training loss ')

    # plt.subplot(212)
    # plt.plot(plot_data["batchindex"], plot_data["avg_error"], 'r--')
    # plt.xlabel('Minibatch number')
    # plt.ylabel('Label Prediction Error')
    # plt.title('Minibatch run vs. Label Prediction Error ')
    # plt.show()
    
    return softmax(z)

import PIL

def eval(pred_op, image_path):
    label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    image_mean   = 133.0
    image_data   = np.array(PIL.Image.open(image_path), dtype=np.float32)
    image_data  -= image_mean
    image_data   = np.ascontiguousarray(np.transpose(image_data, (2, 0, 1)))
    
    result       = np.squeeze(pred_op.eval({pred_op.arguments[0]:[image_data]}))
    
    # Return top 3 results:
    top_count = 3
    result_indices = (-np.array(result)).argsort()[:top_count]

    print("Top 3 predictions:")
    for i in range(top_count):
        print("\tLabel: {:10s}, confidence: {:.2f}%".format(label_lookup[result_indices[i]], result[result_indices[i]] * 100))

def convolution_bn(input, filter_size, num_filters, strides=(1,1), init=he_normal(), activation=relu):
    if activation is None:
        activation = lambda x: x
        
    r = Convolution(filter_size, num_filters, strides=strides, init=init, activation=None, pad=True, bias=False)(input)
    r = BatchNormalization(map_rank=1)(r)
    r = activation(r)
    
    return r

def resnet_basic(input, num_filters):
    c1 = convolution_bn(input, (3,3), num_filters)
    c2 = convolution_bn(c1, (3,3), num_filters, activation=None)
    p  = c2 + input
    return relu(p)

# increase dimension, num_filters = 2 * previous_num_filters
def resnet_basic_inc(input, num_filters):
    c1 = convolution_bn(input, (3,3), num_filters, strides=(2,2))
    c2 = convolution_bn(c1, (3,3), num_filters, activation=None)

    # projection
    s = convolution_bn(input, (1,1), num_filters, strides=(2,2), activation=None)
    
    p = c2 + s
    return relu(p)

def resnet_basic_stack(input, num_filters, num_stack):
    assert (num_stack > 0)
    
    r = input
    for _ in range(num_stack):
        r = resnet_basic(r, num_filters)
    return r

def create_resnet_model(n):
    def func(input, out_dims):
        conv = convolution_bn(input, (3,3), 16)
        r1_1 = resnet_basic_stack(conv, 16, n) # 2n

        r2_1 = resnet_basic_inc(r1_1, 32) # mapping with projection
        r2_2 = resnet_basic_stack(r2_1, 32, n - 1) 
    
        r3_1 = resnet_basic_inc(r2_2, 64)
        r3_2 = resnet_basic_stack(r3_1, 64, n - 1)
    
        # Global average pooling
        pool = AveragePooling(filter_shape=(8,8), strides=(1,1))(r3_2)    
        net = Dense(out_dims, init=he_normal(), activation=None)(pool)
        return net
    func.func_name = "create_resnet_{}".format(n)
    return func

# data_path = os.path.join('data', 'CIFAR-10')
data_path = "/home/foxfi/homework/adavance-ml/Tutorials/data/CIFAR-10"
reader_train = create_reader(os.path.join(data_path, 'train_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), True)
reader_test  = create_reader(os.path.join(data_path, 'test_map.txt'), os.path.join(data_path, 'CIFAR-10_mean.xml'), False)
# resnet-20, n = 3

pred = train_and_evaluate(reader_train, reader_test, max_epochs=200, model_func=create_resnet_model(3))
