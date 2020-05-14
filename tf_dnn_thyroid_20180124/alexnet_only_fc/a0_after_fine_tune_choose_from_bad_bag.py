#!/usr/bin/env python  
#-*- coding: utf-8 -*-

'''
    本脚本用于在训练了全连接网络之后，从bad bag中挑选出初始的用于作为positive的图像
    根据每个bad bag中至少有一个是结节，初始从每个bag中挑选0.3的作为positive图像
'''

'''Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
'''

import os, sys
sys.path.append('../alexnet')
sys.path.append('../alexnet_mil')
import alexnet_mil.utils as utils

import numpy as np
import tensorflow as tf
from alexnet.alexnet import AlexNet
from alexnet.datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator

########################################################################################################################
### 配置：数据、网络结构、训练参数
########################################################################################################################

### 数据
# val_file = 'I:/thyroid-all-gray-20180124/validationData/nodules/validation.txt'
val_file = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules.txt'
result_file = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules_pred_fc.txt'

### 模型位置
checkpoint_path = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath/finetune_alexnet_fc/checkpoints'
checkpoint_path_tmp = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath/finetune_alexnet_fc/checkpoints-tmp'
meta_name = 'model_epoch20.ckpt-3960.meta'
# checkpoint_name = 'model_epoch16.ckpt-3168.data-00000-of-00001'

### 网络结构
img_width = 227; img_height = 227;
batch_size = 128 # 训练的batch_size
num_classes = 2
num_channels = 3
#### 数据
with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size,
                                  num_classes=num_classes,
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
val_batches_per_epoch = np.ceil(val_data.data_size / batch_size).astype(np.int32)

#########
x = tf.placeholder(tf.float32, [batch_size, img_width, img_height, num_channels])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)
model = AlexNet(x, keep_prob, num_classes, skip_layer=[])
score = model.fc8

## 准确率
with tf.name_scope('accuracy'):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    p_pred = tf.nn.softmax(score)

with tf.name_scope('cross_ent'):
    print('lossType: softmax_cross_entropy_loss')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
saver = tf.train.Saver()

predictions = []
with tf.Session() as sess:
    # saver.restore(sess, checkpoint_path + '/' + 'model_epoch20.ckpt-3960')
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

    print('{} Start validation'.format(datetime.now()))
    sess.run(validation_init_op)

    ####################################################################################################################
    ### 开始预测
    ####################################################################################################################
    test_acc = 0.
    test_count = 0
    test_loss = 0.
    for step in range(val_batches_per_epoch):
        print('{} All number: {} Step number: {}'.format(datetime.now(), val_batches_per_epoch, step + 1))
        ###
        img_batch, label_batch = sess.run(next_batch)
        if label_batch.shape[0] < batch_size:
            print('batch not enough, padding')
            print('label.shape = ' + str(label_batch.shape))
            img_batch_new = np.zeros((batch_size, img_width, img_height, num_channels))
            label_batch_new = np.zeros((batch_size, 2))
            img_batch_new[:img_batch.shape[0], :] = img_batch
            label_batch_new[:label_batch.shape[0], :] = label_batch
            img_batch = img_batch_new; label_batch = label_batch_new
            print('img.shape = ' + str(img_batch.shape))

        print('label.shape = ' + str(label_batch.shape))

        [acc, loss_tmp, p_value] = sess.run([accuracy, loss, p_pred],
                                   feed_dict={x: img_batch,
                                              y: label_batch,
                                              keep_prob: 1.0
                                              })
        # print(p_value)
        predictions.append(p_value)
        # exit()
        test_acc += acc
        test_loss += loss_tmp
        test_count += 1

        ####
        # break
        ###########
    test_acc /= test_count
    test_loss /= test_count

    print('{} Validation Accuracy = {:.4f}, Loss = {:.4f}'.format(datetime.now(),
                                                                  test_acc, test_loss))

    # saver.save(sess, checkpoint_path_tmp + '/' + 'crfmodel-test.ckpt', global_step=0)

predictions = np.vstack(predictions)
# print(predictions)
print(predictions.shape)
print('预测完毕，下面将预测结果写入一个文件: ' + result_file)
######## 下面把预测结果写入文件

f = open(val_file)
val_lines = f.readlines()
f.close()

strXieru = ''
for iLine in range(0, len(val_lines)):
    val_line = val_lines[iLine].strip()
    strXieru += val_line + ' ' + str(predictions[iLine][1]) + '\n'

with open(result_file, 'w') as f:
    f.write(strXieru)

print('Done')

####################################
resultPath = 'I:/thyroid-all-gray-20180124/trainData/nodules/choose_for_train_after_fc.txt'
goodPath = 'I:/thyroid-all-gray-20180124/trainData/nodules/good_nodules.txt'
# badPath_predicted = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules_pred_fc.txt'
badPath_predicted = result_file
badDicPath = 'I:/thyroid-all-gray-20180124/trainData/dic_bad.txt'
badDicPath_allBag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_bag_nodules.txt'

nResult = utils.getNextTrainingTxt(resultPath, goodPath, badDicPath, badPath_predicted,
                   badDicPath_allBag, prop=0.5, minCount=1, thresh=0.5)

print('nResult = ' + str(nResult))