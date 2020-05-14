#!/usr/bin/env python  
#-*- coding: utf-8 -*-

########################################################################################################################
########################################################################################################################
#### 本脚本用来预测测试集
########################################################################################################################
########################################################################################################################

import os, sys
sys.path.append('.')

import numpy as np
import tensorflow as tf
from alexnet.alexnet import AlexNet
from alexnet.datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib import slim as slim
import shutil
import alexnet_mil.utils as utils

########################################################################################################################
### 配置：数据、网络结构、训练参数
########################################################################################################################

### 数据
val_file = 'I:/thyroid-all-gray-20180124/testData/nodules/testData-all.txt'
location_file = 'I:/thyroid-all-gray-20180124/testData/nodules/testData-location.txt'
location_file_result = 'I:/thyroid-all-gray-20180124/testData/nodules/testData-location-result-0.1.txt'

############################################
checkpoint_path_fc = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process/finetune_alexnet-20180130/model-1.0-10.0-0.1/checkpoints'
############################################
batch_size_validation = 100

img_width = 227
img_height = 227
class_weight_default = [1, 10] # 默认的class_weight，对于训练数据来讲这个需要改变

### 网络结构
num_classes = 2
num_channels = 3
use_softmax_cross_entropy_loss = True
reinitialization_layers = []  ## 重新初始化变量的层，即不用bvlc_alenenet.npy初始化
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']  ## 会更新的层
########################################################################################################################
## 数据生成
with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size_validation,
                                  num_classes=num_classes,
                                  class_weight=class_weight_default,
                                  shuffle=False)
    ######################################################################
    ##### create an reinitializable iterator given the dataset structure
    ######################################################################
    iterator = Iterator.from_structure(val_data.data.output_types,
                                       val_data.data.output_shapes)
    next_batch = iterator.get_next()

# Ops for initializing the two different iterators
validation_init_op = iterator.make_initializer(val_data.data)
### 计算step
val_batches_per_epoch = val_data.data_size // batch_size_validation
########################################################################################################################
########################################################################################################################
### 一些place holder
x = tf.placeholder(tf.float32, [batch_size_validation, img_width, img_height, num_channels])
y = tf.placeholder(tf.float32, [batch_size_validation, num_classes])
y_weight = tf.placeholder(tf.float32, [batch_size_validation])
keep_prob = tf.placeholder(tf.float32)
# Initialize model
model = AlexNet(x, keep_prob, num_classes, reinitialization_layers)
# Link variable to model output
score = model.fc8
## 准确率
with tf.name_scope('prediction'):
    p_pred = tf.nn.softmax(score)

########################################################################################################################
########################################################################################################################

with tf.Session() as sess:
    ###########################################
    ## 重新初始化
    sess.run(tf.global_variables_initializer())
    ##########
    #### 除了learning rate, global_step, momentum不不从原来的加载进来
    variables = slim.get_variables_to_restore()
    # [print(v) for v in variables]
    # print(variables)
    variables_to_restore = [v for v in variables if v.name.find('Momentum') < 0 and
                            (v.name.find('biases') >= 0 or v.name.find('weights') >= 0)]
    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path_fc))
    ############################################################################################################

    print("{} Start Prediction ...".format(datetime.now()))
    sess.run(validation_init_op)
    #######################################################################
    predictions = []
    for step in range(val_batches_per_epoch):
        # print('{} All number: {} Step number: {}'.format(datetime.now(),val_bad_bag_batches_per_epoch, step + 1))
        img_batch, label_batch, w = sess.run(next_batch)
        if label_batch.shape[0] < batch_size_validation:
            # print('batch not enough, padding')
            # print('label.shape = ' + str(label_batch.shape))
            img_batch_new = np.zeros((batch_size_validation, img_width, img_height, num_channels))
            label_batch_new = np.zeros((batch_size_validation, 2))
            img_batch_new[:img_batch.shape[0], :] = img_batch
            label_batch_new[:label_batch.shape[0], :] = label_batch
            img_batch = img_batch_new;
            label_batch = label_batch_new
            # print('img.shape = ' + str(img_batch.shape))
        # print('label.shape = ' + str(label_batch.shape))

        [p_value] = sess.run([p_pred], feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0 })
        # print(p_value)
        predictions.append(p_value)
    predictions = np.vstack(predictions)
    print(predictions)
    print(predictions.shape)
    ## 写入文件
    if os.path.exists(location_file_result):
        os.remove(location_file_result)
    with open(location_file) as f:
        lines = f.readlines()
    for iline in range(len(lines)):
        lines[iline] = lines[iline].strip().split(' ')[1] +  ',' + str(predictions[iline, 1]) + '\n'
    print(lines)
    with open(location_file_result, 'w') as f:
        f.write(''.join(lines))

print('***********************************************************************************************')
print('Done')