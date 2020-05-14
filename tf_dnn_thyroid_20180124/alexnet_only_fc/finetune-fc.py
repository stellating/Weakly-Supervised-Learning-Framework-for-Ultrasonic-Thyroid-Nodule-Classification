"""Script to finetune AlexNet using Tensorflow.

With this script you can finetune AlexNet as provided in the alexnet.py
class on any given dataset. Specify the configuration settings at the
beginning according to your problem.
This script was written for TensorFlow >= version 1.2rc0 and comes with a blog
post, which you can find here:

https://kratzert.github.io/2017/02/24/finetuning-alexnet-with-tensorflow.html

Author: Frederik Kratzert
contact: f.kratzert(at)gmail.com
"""

import os, sys
sys.path.append('../alexnet')

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
train_file = 'I:/thyroid-all-gray-20180124/trainData/nodules/all_nodules_for_fc.txt'
# train_file = 'I:/thyroid-all-gray-20171213/experiments/open_thyroid_tainAndvalidation/train.txt'
val_file = 'I:/thyroid-all-gray-20180124/validationData/nodules/validation.txt'
# val_file = 'I:/thyroid-all-gray-20171213/experiments/open_thyroid_tainAndvalidation/validation.txt'

### 训练参数
num_epochs = 50 ## 总共最多训练这么多次
base_lr = 0.001 # 最开始的学习率
lr_decay_rate = 0.9
lr_decay_epoch = 1 # 没训练这么多step就重新设置学习率：base_lr *= lr_dacay_rate

batch_size = 64 # 训练的batch_size

img_width = 227; img_height = 227;

### 网络结构
dropout_rate = 0.5
num_classes = 2
num_channels = 3
use_softmax_cross_entropy_loss = True
reinitialization_layers = ['fc8'] ## 重新初始化变量的层，即不用bvlc_alenenet.npy初始化
# train_layers = ['fc8', 'fc7', 'fc6', 'conv5', 'conv4', 'conv3', 'conv2', 'conv1'] ## 会更新的层
train_layers = ['fc8', 'fc7', 'fc6'] ## 会更新的层

### 训练过程
# How often we want to write the tf.summary data to disk
display_step = 20

# Path for tf.summary.FileWriter and to store model checkpoints
filewriter_path = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath/finetune_alexnet_fc/tensorboard'
checkpoint_path = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath/finetune_alexnet_fc/checkpoints'

#
if os.path.exists(checkpoint_path):
    print(checkpoint_path.replace('/', '\\') + ' already exist!!!')
    # exit()
if os.path.exists(filewriter_path):
    print(filewriter_path.replace('/', '\\') + ' already exist!!!')
    # exit()

# Create parent path if it doesn't exist
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(filewriter_path):
    os.makedirs(filewriter_path)

########################################################################################################################
### 训练
########################################################################################################################
## 数据生成
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                 mode='training',
                                 batch_size=batch_size,
                                 num_classes=num_classes,
                                 shuffle=True)
    val_data = ImageDataGenerator(val_file,
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
validation_init_op = iterator.make_initializer(val_data.data)

### 计算step
train_batches_per_epoch = tr_data.data_size // batch_size
val_batches_per_epoch = val_data.data_size // batch_size

### 一些place holder
x = tf.placeholder(tf.float32, [batch_size, img_width, img_height, num_channels])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
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
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                                                                  labels=y))
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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
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
writer = tf.summary.FileWriter(filewriter_path)
saver = tf.train.Saver()

########################################################################################################################
### 开始session过程
########################################################################################################################
with tf.Session() as sess:
    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    # Add the model graph to TensorBoard
    writer.add_graph(sess.graph)
    ##### Load the pretrained weights into the non-trainable layer
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                      filewriter_path))

    ####################################################################################################################
    ### 开始训练
    ####################################################################################################################
    for epoch in range(num_epochs):

        print("{} Epoch number: {}".format(datetime.now(), epoch+1))

        ### Initialize iterator with the training dataset
        sess.run(training_init_op)

        for step in range(train_batches_per_epoch):

            ### 获取下一个batch的数据
            img_batch, label_batch = sess.run(next_batch)

            ### 训练当前batch的数据
            sess.run(train_op, feed_dict={x: img_batch,
                                          y: label_batch,
                                          keep_prob: dropout_rate})

            ### 讲summary信息保存
            if step % display_step == 0:
                s = sess.run(merged_summary,
                             feed_dict={x: img_batch,
                                        y: label_batch,
                                        keep_prob: 1.
                                        })

                writer.add_summary(s, epoch*train_batches_per_epoch + step)

            ############################################################################################################
            ## end for : step (batch)

        ### 在验证集上面验证
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
                img_batch = img_batch_new;
                label_batch = label_batch_new
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

        print("{} Saving checkpoint of model...".format(datetime.now()))

        ### 保存checkpoint
        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
        save_path = saver.save(sess, checkpoint_name, global_step=global_step)
        print('save_path = ' + save_path)

        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))

    ####################################################################################################################
    ## end for : epoch