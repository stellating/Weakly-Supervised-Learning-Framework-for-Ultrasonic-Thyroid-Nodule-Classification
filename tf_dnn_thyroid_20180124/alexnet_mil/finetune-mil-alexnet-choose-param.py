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
train_file_after_fc = 'I:/thyroid-all-gray-20180124/trainData/nodules/choose_for_train_after_fc.txt'
val_file_good = 'I:/thyroid-all-gray-20180124/validationData/nodules/validationData-good.txt'
val_file_bad = 'I:/thyroid-all-gray-20180124/validationData/nodules/validationData-bad.txt'

val_file_bad_bag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules.txt'

############################################
badPath_predicted = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules_pred_mil.txt'
goodPath = 'I:/thyroid-all-gray-20180124/trainData/nodules/good_nodules.txt'
badDicPath = 'I:/thyroid-all-gray-20180124/trainData/dic_bad.txt'
badPath = val_file_bad_bag
badDicPath_allBag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_bag_nodules.txt'
checkpoint_path_fc = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath/finetune_alexnet_fc/checkpoints'
############################################
num_epochs = 30  # 总共最多训练这么多次
momentum = 0.9
base_lr = 0.001  # 最开始的学习率
lr_decay_rate = 0.9
lr_decay_epoch = 1  # 没训练这么多step就重新设置学习率：base_lr *= lr_dacay_rate

batch_size = 64  # 训练的batch_size
batch_size_validation = batch_size  # 测试的batch_size，必须一样

img_width = 227;
img_height = 227

### 网络结构
dropout_rate = 0.5
num_classes = 2
num_channels = 3
use_softmax_cross_entropy_loss = True
reinitialization_layers = []  ## 重新初始化变量的层，即不用bvlc_alenenet.npy初始化
train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1']  ## 会更新的层
# train_layers = ['fc8', 'fc7', 'fc6'] ## 会更新的层

display_step = 20

###
rootDir_path = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process/finetune_alexnet'
if os.path.exists((rootDir_path)):
    print(rootDir_path + ' alrady exist!!!')
    # exit()
########################################################################################################################
weightPositives = [3.0, 5.0, 1.0]
propPositives = [0.3, 0.5, 0.7]
########################################################################################################################
for iWeight in range(0, len(weightPositives)):
    weightPositive = weightPositives[iWeight]
    for iProp in range(0, len(propPositives)):
        propPositive = propPositives[iProp]
        ################################################################################################################
        propP = propPositive
        weight_class = [1.0, weightPositive]
        print('weightP = ' + str(weight_class) + ', proP = ' + str(propP))

        ### 训练参数
        dataDirThis = rootDir_path + '/model-' \
                      + str(weightPositive) + '-' + str(propPositive)

        if not os.path.exists(dataDirThis):
            os.makedirs(dataDirThis)
        train_file = dataDirThis + '/' + 'train.txt'
        log_validation_data = dataDirThis + '/' + 'log_validation_Data.txt'

        if os.path.exists(train_file):
            os.remove(train_file)
        shutil.copy(train_file_after_fc, train_file)
        if os.path.exists(log_validation_data):
            os.remove(log_validation_data)
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

        ########################################################################################################################
        ### 训练
        ########################################################################################################################
        print('数据生成...')
        ## 数据生成
        with tf.device('/cpu:0'):
            tr_data = ImageDataGenerator(train_file,
                                         mode='training',
                                         batch_size=batch_size,
                                         num_classes=num_classes,
                                         shuffle=True)
            val_data_good = ImageDataGenerator(val_file_good,
                                          mode='inference',
                                          batch_size=batch_size,
                                          num_classes=num_classes,
                                          shuffle=False)
            val_data_bad = ImageDataGenerator(val_file_bad,
                                          mode='inference',
                                          batch_size=batch_size,
                                          num_classes=num_classes,
                                          shuffle=False)
            val_data_bad_bag = ImageDataGenerator(val_file_bad_bag,
                                          mode='inference',
                                          batch_size=batch_size,
                                          num_classes=num_classes,
                                          shuffle=False)

            ######################################################################
            ##### create an reinitializable iterator given the dataset structure
            ######################################################################
            iterator = Iterator.from_structure(tr_data.data.output_types,
                                               tr_data.data.output_shapes)
            next_batch = iterator.get_next()

        # Ops for initializing the two different iterators
        training_init_op = iterator.make_initializer(tr_data.data)
        validation_good_init_op = iterator.make_initializer(val_data_good.data)
        validation_bad_init_op = iterator.make_initializer(val_data_bad.data)
        validation_bad_bag_init_op = iterator.make_initializer(val_data_bad_bag.data)

        ### 计算step
        train_batches_per_epoch = tr_data.data_size // batch_size
        val_good_batches_per_epoch = val_data_good.data_size // batch_size_validation
        val_bad_batches_per_epoch = val_data_bad.data_size // batch_size_validation
        val_bad_bag_batches_per_epoch = np.ceil(val_data_bad_bag.data_size / batch_size_validation).astype(np.int32)

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

        with tf.name_scope('my'):
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
        #
        # for var in var_list:
        #     tf.summary.histogram(var.name, var)

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
        writer = tf.summary.FileWriter(filewriter_path)
        ########################################################################################################################
        ### 开始session过程
        ########################################################################################################################
        with tf.Session() as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # Add the model graph to TensorBoard
            writer.add_graph(sess.graph)
            ##########
            #### 除了learning rate, global_step, momentum不不从原来的加载进来
            variables = slim.get_variables_to_restore()
            # [print(v) for v in variables]
            # print(variables)
            variables_to_restore = [v for v in variables if v.name.find('Momentum') < 0 and
                                    (v.name.find('biases') >= 0 or v.name.find('weights') >= 0)]
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path_fc))
            ##########

            print("{} Start training...".format(datetime.now()))
            print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                              filewriter_path))

            print('learning rate = ' + str(sess.run(learning_rate)))
            print('global step = ' + str(sess.run(global_step)))

            ####################################################################################################################
            ### 开始训练
            ####################################################################################################################
            for epoch in range(num_epochs):
                print('weightP = ' + str(weight_class) + ', proP = ' + str(propP))
                print("{} Epoch number: {}".format(datetime.now(), epoch+1))
                ### Initialize iterator with the training dataset
                sess.run(training_init_op)
                for step in range(train_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    w = np.ones(img_batch.shape[0], dtype=np.float32)
                    for ilabel in range(label_batch.shape[0]):
                        if label_batch[ilabel][0] > 0.5: ## 属于第0类
                            w[ilabel] = weight_class[0]
                        else: ## 属于第1类
                            w[ilabel] = weight_class[1]

                    [train_op1, score1] = sess.run([train_op, score], feed_dict={x: img_batch, y_weight: w,
                                                                                 y: label_batch, keep_prob: dropout_rate })
                    # print(score1.shape)
                    # exit()
                    ### 将summary信息保存
                    if step % display_step == 0:
                        s = sess.run(merged_summary, feed_dict={x: img_batch, y_weight: w, y: label_batch, keep_prob: 1. })
                        writer.add_summary(s, epoch*train_batches_per_epoch + step)

                    ## end for : step (batch)

                ################################################################################################################
                ################################################################################################################
                ################################################################################################################
                ### 在验证集上面验证 -- good
                w = np.ones(batch_size_validation, dtype=np.float32)
                print("{} Start validation".format(datetime.now()))
                sess.run(validation_good_init_op)
                test_acc = 0.; test_count = 0; test_loss = 0.
                for _ in range(val_good_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    [acc, loss_tmp] = sess.run([accuracy, loss],
                                               feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: 1. })
                    test_acc += acc; test_loss += loss_tmp; test_count += 1
                test_acc /= test_count; test_loss /= test_count
                print("{} Validation Good Accuracy = {:.4f}, Loss = {:.4f}".format(datetime.now(),
                                                               test_acc, test_loss))
                good_loss = test_loss; good_acc = test_acc;
                ### 在验证集上面验证 -- bad
                print("{} Start validation".format(datetime.now()))
                sess.run(validation_bad_init_op)
                test_acc = 0.; test_count = 0; test_loss = 0.
                for _ in range(val_bad_batches_per_epoch):
                    img_batch, label_batch = sess.run(next_batch)
                    [acc, loss_tmp] = sess.run([accuracy, loss],
                                               feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: 1. })
                    test_acc += acc; test_loss += loss_tmp; test_count += 1
                test_acc /= test_count; test_loss /= test_count
                print("{} Validation Bad Accuracy = {:.4f}, Loss = {:.4f}".format(datetime.now(),
                                                                                   test_acc, test_loss))
                bad_loss = test_loss; bad_acc = test_acc;
                #########
                with open(log_validation_data, 'a') as f:
                    f.write(str(epoch + 1) + ', ' + str(good_loss) + ', ' + str(bad_loss) + ', '
                            + str(good_acc) + ', ' + str(bad_acc) + ', '
                            + str((good_loss + bad_loss) / 2.) + ', ' + str((good_acc + bad_acc) / 2.) + '\n'
                            )
                ################################################################################################################
                ################################################################################################################
                ################################################################################################################
                print("{} Saving checkpoint of model...".format(datetime.now()))
                ### 保存checkpoint
                checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                save_path = saver.save(sess, checkpoint_name, global_step=global_step)
                print('save_path = ' + save_path)
                print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

                ################################################################################################################
                ################################################################################################################
                ################################################################################################################
                ### 下面重新预测bad bag并修改以进行下一轮迭代
                print('{} Start validation bad bag'.format(datetime.now()))
                sess.run(validation_bad_bag_init_op)
                ####################################################################################################################
                ### 开始预测
                ####################################################################################################################
                test_acc = 0.; test_count = 0; test_loss = 0.; predictions = []
                for step in range(val_bad_bag_batches_per_epoch):
                    # print('{} All number: {} Step number: {}'.format(datetime.now(),val_bad_bag_batches_per_epoch, step + 1))
                    img_batch, label_batch = sess.run(next_batch)
                    if label_batch.shape[0] < batch_size:
                        # print('batch not enough, padding')
                        # print('label.shape = ' + str(label_batch.shape))
                        img_batch_new = np.zeros((batch_size, img_width, img_height, num_channels))
                        label_batch_new = np.zeros((batch_size, 2))
                        img_batch_new[:img_batch.shape[0], :] = img_batch
                        label_batch_new[:label_batch.shape[0], :] = label_batch
                        img_batch = img_batch_new;
                        label_batch = label_batch_new
                        # print('img.shape = ' + str(img_batch.shape))
                    # print('label.shape = ' + str(label_batch.shape))

                    [p_value] = sess.run([p_pred], feed_dict={x: img_batch, y: label_batch, y_weight: w, keep_prob: 1.0 })
                    # print(p_value)
                    predictions.append(p_value)

                ################################################################################################################
                ### 下面写入文件
                predictions = np.vstack(predictions)

                utils.writePrediction(badPath_predicted, badPath, predictions)
                nResult = utils.getNextTrainingTxt(train_file, goodPath, badDicPath, badPath_predicted,
                                                   badDicPath_allBag, prop=propP, minCount=1, thresh=0.5)

                ## 修改训练数据，下面的函数是自己写的，每次重新读取文件，重新设置data set
                print('nResult = ' + str(nResult))
                tr_data.read_txt_file()
                tr_data.resetData()

                ##
                # break ## end epoch
        ###################################################################################################################
        ####################################################################################################################