#!/usr/bin/env python  
#-*- coding: utf-8 -*-

'''
    本脚本用来跑adaptive选择正样本的代码
    主要有一些参数
'''

#
import os, sys
sys.path.append('.')
#
import numpy as np
import tensorflow as tf
from alexnet.alexnet import AlexNet
from alexnet.datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib import slim as slim
import alexnet_mil_adaptive_20181117.utils as utils

########################################################################################################################
### 参数
Cs = [5] ### 表示总共训练次数
Ts = [10] ### 表示每次训练多少epoch
TTT_LAST = 10

count_good = 1778 ### is for data augmentation
Counts = [[1, 1], [2, 2], [3, 3], [2, 1], [3, 1], [3, 2]] ## 表示数量下降的最高值和最低值
##
weightPositive = 1
weightNegative = 1
thresh_easy = 0.99
##
# model_save_dir = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process_20181116/finetune_alexnet_bishe_20181116' \
#                + '_wp_' + str(weightPositive) + '_te_' + str(thresh_easy)
##
model_save_dir = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process_20181116/finetune_alexnet_bishe_20181116_wp_1_te_0.99'
res_dir = 'I:/thyroid-all-gray-20180124/validationData/nodules/all_experiments_res' \
          + '_' + str(weightPositive) + '_' + str(thresh_easy)
##
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
##
val_file = 'I:/thyroid-all-gray-20180124/validationData/nodules/validationData-all.txt'
##
propP = 0.0 ## 每个bag选择预测值最高的那一个
thresh_negative = 0.4
###
momentum = 0.9
base_lr = 0.0001 # 最开始的学习率
lr_decay_rate = 0.5
lr_decay_epoch = 1 # 没训练这么多step就重新设置学习率：base_lr *= lr_dacay_rate

batch_size = 200 # 训练的batch_size
batch_size_validation = batch_size # 测试的batch_size，必须一样
###########
img_width = 227; img_height = 227
#
### 网络结构
dropout_rate = 0.5
num_classes = 2
num_channels = 3
weight_class = [weightNegative, weightPositive]
use_softmax_cross_entropy_loss = True
reinitialization_layers = [] ## 重新初始化变量的层，即不用bvlc_alenenet.npy初始化
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1'] ## 会更新的层

### 训练过程
# How often we want to write the tf.summary data to disk
display_step = 20
##

########################################################################################################################
########################################################################################################################
## 训练
########################################################################################################################
## 数据生成
with tf.device('/cpu:0'):
    val_data = ImageDataGenerator(val_file,
                                  mode='inference',
                                  batch_size=batch_size_validation,
                                  num_classes=num_classes,
                                  class_weight=[1, 1],
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
val_batches_per_epoch = np.ceil(val_data.data_size / batch_size_validation).astype(np.int32)

########################################################################################################################
### 一些place holder
x = tf.placeholder(tf.float32, [batch_size, img_width, img_height, num_channels])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
y_weight = tf.placeholder(tf.float32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
with tf.name_scope('build_model'):
    model = AlexNet(x, keep_prob, num_classes, reinitialization_layers)
    # Link variable to model output
    score = model.fc8

########################################################################################################################
for cccount in Counts: ### 对于每种数量组合
    max_count = cccount[0] * count_good; min_count = cccount[1] * count_good
    model_save_dir_cccount = os.path.join(model_save_dir, 'experiment-count' + '-' + str(max_count) + '-' + str(min_count))
    if not os.path.exists(model_save_dir_cccount):
        os.mkdir(model_save_dir_cccount)
    for ccc in Cs: ### 对于每种训练次数

        model_save_dir_ccc = os.path.join(model_save_dir_cccount, 'ctimes-' + str(ccc))
        if not os.path.exists(model_save_dir_ccc):
            os.mkdir(model_save_dir_ccc)
        ##
        training_imgs_good_plus_bag = (max_count + count_good) * 3 ####### * 3 is for data augmentation
        print('training_imgs_good_plus_bag = ', training_imgs_good_plus_bag)
        ## 对于每一次训练
        for ittt in Ts: ### 对于每种迭代epoch
            ##
            #####################################
            for iccc in range(ccc): ## 对于每一次训练
                ###
                if iccc == ccc - 1:
                    ttt = TTT_LAST
                else:
                    ttt = ittt
                ###
                print()
                print('######################################################################################')
                print('######################################################################################')
                print('cccount = ', cccount, 'ccc = ', ccc, ', ttt = ', ttt, ' ----------------- iccc = ', iccc)
                #
                location_file_result = os.path.join(res_dir, 'validationData-all' + '-' + str(cccount[0]) + '-' + str(cccount[1])+ '-' + 'c' + '-' + str(iccc + 1)  + '.txt')
                #
                model_save_dir_iccc = os.path.join(model_save_dir_ccc, 'iccc-' + str(iccc))
                #
                checkpoint_path = os.path.join(model_save_dir_iccc, 'checkpoints')
                #
                print('checkpoint_path = ', checkpoint_path)
                ###
                ########################################################################################################
                ## *****************************************************************************************************

                ### 计算step
                val_batches_per_epoch = np.ceil(val_data.data_size / batch_size_validation).astype(np.int32)

                ## 准确率
                with tf.name_scope("accuracy"):
                    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                    p_pred = tf.nn.softmax(score)
                ##
                ########################################################################################################################
                ### 开始session过程
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
                    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
                    ############################################################################################################

                    print("{} Start Prediction ...".format(datetime.now()))
                    sess.run(validation_init_op)
                    #######################################################################
                    predictions = []
                    for step in range(val_batches_per_epoch):
                        img_batch, label_batch, w = sess.run(next_batch)
                        print(
                            '{} All number: {} Step number: {}'.format(datetime.now(), val_batches_per_epoch, step + 1))
                        print('\tsize = ' + str(label_batch.shape[0]))
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

                        [p_value] = sess.run([p_pred], feed_dict={x: img_batch, y: label_batch, keep_prob: 1.0})
                        # print(p_value)
                        predictions.append(p_value)
                    predictions = np.vstack(predictions)
                    # print(predictions)
                    print(predictions.shape)
                    ## 写入文件
                    with open(val_file) as f:
                        lines = f.readlines()
                    for iline in range(len(lines)):
                        # lines[iline] = lines[iline].strip() +  ' ' + str(predictions[iline, 1]) + '\n'
                        lines[iline] = lines[iline].strip().split(' ')[1] + ',' + str(predictions[iline, 1]) + '\n'
                    print(lines)
                    with open(location_file_result, 'w') as f:
                        f.write(''.join(lines))
                ###
            ## end ttt
        ## end 每一次
    ### end ccc
## end for cccount

print('Done')
print('res_path = ', model_save_dir)