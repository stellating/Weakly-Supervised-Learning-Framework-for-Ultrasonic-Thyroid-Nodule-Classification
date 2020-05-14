#!/usr/bin/env python  
#-*- coding: utf-8 -*-

########################################################################################################################
########################################################################################################################
##    本脚本用来跑mil版本的程序
##   在fully-connected的基础上进行
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
train_file = 'I:/thyroid-all-gray-20180124/openData-finetune-classification-60-80/train/nodules/all_nodules.txt'
val_file_good = 'I:/thyroid-all-gray-20180124/openData-finetune-classification-60-80/validation/nodules-1/good_nodules.txt'
val_file_bad = 'I:/thyroid-all-gray-20180124/openData-finetune-classification-60-80/validation/nodules-1/bad_nodules.txt'
############################################
rootDir_path = 'I:/thyroid-all-gray-20180124/openData-finetune-classification-60-80/train-process/finetune_alexnet_on_opendatabase'
validationPath = 'I:/thyroid-all-gray-20180124/openData-finetune-classification-60-80/validation'
############################################
num_epochs = 200  # 总共最多训练这么多次
momentum = 0.9
base_lr = 0.00005  # 最开始的学习率
lr_decay_rate = 0.96
lr_decay_epoch = 2  # 没训练这么多step就重新设置学习率：base_lr *= lr_dacay_rate

batch_size = 32  # 训练的batch_size
batch_size_validation_good = 32  # 测试的batch_size，必须一样
batch_size_validation_bad = 32  # 测试的batch_size，必须一样

img_width = 227
img_height = 227
class_weight_default = [1, 1] # 默认的class_weight，对于训练数据来讲这个需要改变

### 网络结构
dropout_rate = 0.5
num_classes = 2
num_channels = 3
use_softmax_cross_entropy_loss = True
reinitialization_layers = ['fc8']  ## 重新初始化变量的层，即不用bvlc_alenenet.npy初始化
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']  ## 会更新的层
# train_layers = ['fc8', 'fc7', 'fc6'] ## 会更新的层
display_step = 1

########################################################################################################################
########################################################################################################################
## 数据生成
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 class_weight=class_weight_default,
                                 shuffle=True)
    val_data_good = ImageDataGenerator(val_file_good,
                                  mode='inference',
                                  batch_size=batch_size_validation_good,
                                  num_classes=num_classes,
                                  class_weight=class_weight_default,
                                  shuffle=False)
    val_data_bad = ImageDataGenerator(val_file_bad,
                                  mode='inference',
                                  batch_size=batch_size_validation_bad,
                                  num_classes=num_classes,
                                  class_weight=class_weight_default,
                                  shuffle=False)

    ######################################################################
    ##### create an reinitializable iterator given the dataset structure
    ######################################################################
    iterator = Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
    next_batch = iterator.get_next()

    iterator_validation_good = Iterator.from_structure(val_data_good.data.output_types,
                                                  val_data_good.data.output_shapes)
    next_batch_validation_good = iterator_validation_good.get_next()

    iterator_validation_bad = Iterator.from_structure(val_data_good.data.output_types,
                                                   val_data_good.data.output_shapes)
    next_batch_validation_bad = iterator_validation_bad.get_next()

# Ops for initializing the two different iterators
training_init_op = iterator.make_initializer(tr_data.data)
# validation_good_init_op = iterator_validation_good.make_initializer(val_data_good.data)
# validation_bad_init_op = iterator_validation_bad.make_initializer(val_data_bad.data)

### 计算step
train_batches_per_epoch = tr_data.data_size // batch_size

########################################################################################################################
########################################################################################################################
########################################################################################################################
### 一些place holder
x = tf.placeholder(tf.float32, [batch_size, img_width, img_height, num_channels])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
y_weight = tf.placeholder(tf.float32, [batch_size])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, reinitialization_layers, weights_path = '../alexnet/bvlc_alexnet.npy')

# Link variable to model output
score = model.fc8

# List of trainable variables of the layers we want to train
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

# Op for calculating the loss
with tf.name_scope("cross_ent"):
    if num_classes > 2 or use_softmax_cross_entropy_loss:
        print('lossType: softmax_cross_entropy_loss')
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
        #                                                           labels=y))

        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y, logits=score, weights=y_weight))

    elif num_classes == 2:
        print('lossType: sigmoid_cross_entropy_loss')
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
    else:
        print('wrong num_classes!!!')
        exit()

##
global_step = tf.Variable(0, name='global_step', trainable=False)

learning_rate = tf.train.exponential_decay(base_lr, global_step,
                                           lr_decay_epoch * train_batches_per_epoch, lr_decay_rate, staircase=True)
# Train op
with tf.name_scope("train"):
    # Get gradients of all trainable variables
    gradients = tf.gradients(loss, var_list)
    gradients = list(zip(gradients, var_list))

    # Create optimizer and apply gradient descent to the trainable variables
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
    train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)

### 添加到summary
for gradient, var in gradients:
    tf.summary.histogram(var.name + '/gradient', gradient)

for var in var_list:
    tf.summary.histogram(var.name, var)

## 准确率
with tf.name_scope("accuracy"):
    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    p_pred = tf.nn.softmax(score)

### 添加到summary
tf.summary.scalar('train_accuracy', accuracy)
tf.summary.scalar('train_loss', loss)
tf.summary.scalar('learning_rate', learning_rate)

# Merge all summaries together
merged_summary = tf.summary.merge_all()

########################################################################################################################
########################################################################################################################
### 开始循环
if os.path.exists((rootDir_path)):
    print(rootDir_path + ' alrady exist!!!')
    # exit()
########################################################################################################################
weightPositives = [1.0]
validationDirs = ['1', '1.1', '1.2']
########################################################################################################################
for iWeight in range(0, len(weightPositives)):
    weightPositive = weightPositives[iWeight]

    ################################################################################################################
    weight_class = [1.0, weightPositive]
    print('****************************************************************************************')
    print('weightP = ' + str(weight_class))
    print('****************************************************************************************')
    #######################################################################
    ### 训练参数
    dataDirThis = rootDir_path + '/model-original-alexnet-' \
                  + str(weight_class[0]) + '-'+ str(weight_class[1]) + '-' \
                  + str(base_lr) + '-' + str(lr_decay_rate) + '-' + str(lr_decay_epoch)
    predictionPath = dataDirThis + '/prediction'

    sum_weight = (weight_class[0] + weight_class[1])
    weight_class[0] = weight_class[0] / sum_weight
    weight_class[1] = weight_class[1] / sum_weight

    if not os.path.exists(dataDirThis):
        os.makedirs(dataDirThis)
    if not os.path.exists(predictionPath):
        os.makedirs(predictionPath)
    log_validation_data = dataDirThis + '/' + 'log_validation_Data.txt'

    with open(log_validation_data, 'w') as f:
        f.write('')

    # Path for tf.summary.FileWriter and to store model checkpoints
    filewriter_path = dataDirThis + '/tensorboard'
    checkpoint_path = dataDirThis + '/checkpoints'

    # Create parent path if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.isdir(filewriter_path):
        os.makedirs(filewriter_path)

    #######################################################################
    writer = tf.summary.FileWriter(filewriter_path)
    with tf.Session() as sess:
        ###########################################
        ## 重新初始化
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)
        model.load_initial_weights(sess)
        ##########
        #### 除了learning rate, global_step, momentum不不从原来的加载进来

        print('learning_rate = ' + str(sess.run(learning_rate)))
        print('global_step = ' + str(sess.run(global_step)))
        ############################################################################################################

        print("{} Start training...".format(datetime.now()))
        print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))
        #######################################################################
        ## 重置一些变量，这个是重置读取数据的
        # tr_data.set_class_weight(weight_class)
        # tr_data.set_txt_file(train_file)
        # tr_data.read_txt_file()
        # tr_data.resetData()
        ##
        ####################################################################################################################
        ### 开始训练
        ####################################################################################################################
        for epoch in range(num_epochs):
            print('****************************************************************************************')
            print('weightP = ' + str(weight_class))
            print('****************************************************************************************')
            print("{} All Epoch: {}, Epoch number: {}".format(datetime.now(), num_epochs, epoch + 1))
            ### Initialize iterator with the training dataset
            print('tr_data.data_size = ' + str(tr_data.data_size))
            train_batches_per_epoch = tr_data.data_size // batch_size
            training_init_op = iterator.make_initializer(tr_data.data)
            sess.run(training_init_op)
            for step in range(train_batches_per_epoch):
                img_batch, label_batch, w = sess.run(next_batch)
                # print(label_batch)
                # print(w)
                sess.run(train_op, feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: dropout_rate })
                ### 将summary信息保存
                if step % display_step == 0:
                    ## 此时weight置为相等
                    w = np.ones(img_batch.shape[0], dtype=np.float32)
                    s = sess.run(merged_summary, feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: 1. })
                    writer.add_summary(s, epoch*train_batches_per_epoch + step)

                    # break
                # break

                ## end for : step (batch)

            ################################################################################################################
            ################################################################################################################
            ################################################################################################################
            ## 预测
            strXieru = str(epoch)
            print("{} Start validation".format(datetime.now()))
            for validationDirName in validationDirs:
                validationPath_this = validationPath + '/' + 'nodules-' + validationDirName
                val_good = validationPath_this + '/' + 'good_nodules.txt'
                val_bad = validationPath_this + '/' + 'bad_nodules.txt'

                ########################################################################################################
                ### 验证 good
                ###############
                val_data_good.set_txt_file(val_good)
                val_data_good.read_txt_file()
                val_data_good.resetData()
                validation_batches_per_epoch_good = np.ceil(val_data_good.data_size / batch_size_validation_good).astype(np.int32)
                validation_good_init_op = iterator_validation_good.make_initializer(val_data_good.data)
                sess.run(validation_good_init_op)
                ###############
                right_count = 0;
                test_count = 0;
                predictions = []
                labels = []
                for _ in range(validation_batches_per_epoch_good):
                    img_batch, label_batch, w = sess.run(next_batch_validation_good)
                    # print(label_batch)
                    count_this = img_batch.shape[0]
                    if count_this < batch_size:
                        # print('batch not enough, padding')
                        # print('label.shape = ' + str(label_batch.shape))
                        img_batch_new = np.zeros((batch_size, img_width, img_height, num_channels))
                        label_batch_new = np.zeros((batch_size, 2))
                        img_batch_new[:img_batch.shape[0], :] = img_batch
                        label_batch_new[:label_batch.shape[0], :] = label_batch
                        img_batch = img_batch_new;
                        label_batch = label_batch_new
                        # print('img.shape = ' + str(img_batch.shape))
                    w = np.ones(img_batch.shape[0], dtype=np.float32)
                    [p_value] = sess.run([p_pred],
                                               feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: 1.})
                    p_value = p_value[:count_this, :]
                    predictions.append(p_value)
                    # print(p_value)
                    n_this = p_value.shape[0] * 1.
                    labels.append(np.zeros(np.int32(n_this), dtype=np.int32))

                    test_count += n_this
                    right_count += np.count_nonzero(p_value[:,0] > 0.5)

                test_acc = right_count * 1.0 / test_count
                strXieru += ', ' + str(test_acc)
                print("{} All count {}, Right count = {}, Validation Good Accuracy = {:.4f}"
                      .format(datetime.now(), test_count, right_count, test_acc))

                ########################################################################################################
                ### 验证 good
                ###############
                val_data_bad.set_txt_file(val_bad)
                val_data_bad.read_txt_file()
                val_data_bad.resetData()
                validation_batches_per_epoch_bad = np.ceil(val_data_bad.data_size / batch_size_validation_bad).astype(np.int32)
                validation_bad_init_op = iterator_validation_bad.make_initializer(val_data_bad.data)
                sess.run(validation_bad_init_op)
                ###############
                right_count = 0;
                test_count = 0;
                for _ in range(validation_batches_per_epoch_bad):
                    img_batch, label_batch, w = sess.run(next_batch_validation_bad)
                    # print(label_batch)
                    count_this = img_batch.shape[0]
                    if count_this < batch_size:
                        # print('batch not enough, padding')
                        # print('label.shape = ' + str(label_batch.shape))
                        img_batch_new = np.zeros((batch_size, img_width, img_height, num_channels))
                        label_batch_new = np.zeros((batch_size, 2))
                        img_batch_new[:img_batch.shape[0], :] = img_batch
                        label_batch_new[:label_batch.shape[0], :] = label_batch
                        img_batch = img_batch_new;
                        label_batch = label_batch_new
                        # print('img.shape = ' + str(img_batch.shape))
                    w = np.ones(img_batch.shape[0], dtype=np.float32)
                    [p_value] = sess.run([p_pred],
                                         feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: 1.})
                    p_value = p_value[:count_this, :]
                    predictions.append(p_value)
                    # print(p_value)
                    n_this = p_value.shape[0] * 1.
                    labels.append(np.ones(np.int32(n_this), dtype=np.int32))

                    test_count += n_this
                    right_count += np.count_nonzero(p_value[:, 1] > 0.5)

                test_acc = right_count * 1.0 / test_count
                strXieru += ', ' + str(test_acc)
                print("{} All count {}, Right count = {}, Validation Bad Accuracy = {:.4f}"
                      .format(datetime.now(), test_count, right_count, test_acc))

                if validationDirName == '1':
                    predictions = np.vstack(predictions)
                    labels = np.hstack(labels)

                    print(predictions.shape)
                    print(labels.shape)
                    strXieruTmp = ''
                    for iline in range(0, predictions.shape[0]):
                        strXieruTmp += str(labels[iline]) + ', ' + str(predictions[iline][1]) + '\n'
                    # exit()
                    with open(predictionPath + '/' + str(epoch) + '.txt', 'w') as f:
                        f.write(strXieruTmp)
            ##
            strXieru += '\n'
            with open(log_validation_data, 'a') as f:
                f.write(strXieru)
            ################################################################################################################
            ################################################################################################################
        ####################################################################################################################
        ## end for : epoch


print('***********************************************************************************************')
print('Done')