#!/usr/bin/env python  
#-*- coding: utf-8 -*-
########################################################################################################################
########################################################################################################################
#### 本脚本用来预测测试集
########################################################################################################################
########################################################################################################################

import os, sys, time
sys.path.append('.')

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
from alexnet.alexnet import AlexNet
from alexnet.datagenerator import ImageDataGenerator
from datetime import datetime
from tensorflow.contrib.data import Iterator
from tensorflow.contrib import slim as slim
#############################################


########################################################################################################################
### 配置：数据、网络结构、训练参数
########################################################################################################################

### 数据
Counts11 = [[1, 1], [2, 2], [3, 3], [2, 1], [3, 1], [3, 2]] ## 表示数量下降的最高值和最低值
Counts11 = [[3, 2]] ## 表示数量下降的最高值和最低值
Counts = Counts11[0] ## 表示数量下降的最高值和最低值

##
val_file = 'D:/work_space/pycharm_keras/EndMaster_Demo/nodule_path_init.txt'

############################################
count_good = 1778 ### is for data augmentation
#
C = 5
T = 0
model_save_dir = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process_20181116/finetune_alexnet_bishe_20181116_wp_1_te_0.99'
# model_save_dir = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process_20181116/finetune_alexnet_bishe_20181116_wp_1_te_1.1'
max_count = Counts[0] * count_good; min_count = Counts[1] * count_good
# max_count = 500; min_count = 500;
model_save_dir_cccount = os.path.join(model_save_dir, 'experiment-count' + '-' + str(max_count) + '-' + str(min_count))
model_save_dir_ccc = os.path.join(model_save_dir_cccount, 'ctimes-' + str(C))
model_save_dir_iccc = os.path.join(model_save_dir_ccc, 'iccc-' + str(T))
checkpoint_path = os.path.join(model_save_dir_iccc, 'checkpoints')
print('checkpoint_path = ', checkpoint_path)
### checkpoint_path = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath-20181116/finetune_alexnet_fc_5.0/checkpoints'
############################################
batch_size_validation = 32

img_width = 227
img_height = 227
class_weight_default = [1, 1] # 默认的class_weight，对于训练数据来讲这个需要改变

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

########################################################################################################################
 # Ops for initializing the two different iterators
validation_init_op = iterator.make_initializer(val_data.data)
### 计算step
val_batches_per_epoch = np.ceil(val_data.data_size / batch_size_validation).astype(np.int32)

print('all_data = ' + str(val_data.data_size) + '\t' + 'batch_size = ' + str(batch_size_validation) + '\t'
      + 'all_steps = ' + str(val_batches_per_epoch))
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

print('start predict:')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    #with tf.device("/cpu:0"):
    if 2 > 1:

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


        ############################################################################################################
        import socket  # 导入 socket 模块

        i_start = 0
        while True:
            print()
            print('**********************************');
            print('count = ', i_start + 1)

            i_start += 1
            #
            s = socket.socket()  # 创建 socket 对象
            host = '219.224.167.239'  # 获取本地主机名
            port = 19999  # 设置端口
            s.bind((host, port))  # 绑定端口
            s.listen(5)  # 等待客户端连接
            print('receiving ...')

            c, addr = s.accept()  # 建立客户端连接。
            data = c.recv(2048)
            data = data.decode("utf-8")
            print(data)
            ##
            strs = data.split(';;;');
            val_file = strs[0]
            location_file_result = strs[1]
            ############################################################################################################
            print("{} Start Prediction ...".format(datetime.now()))
            print('更新预测的数据')
            val_data.set_txt_file(val_file)
            val_data.read_txt_file()
            val_data.resetData()
            sess.run(validation_init_op)
            #######################################################################
            predictions = []
            start = time.clock()
            print('开始预测')
            for step in range(val_batches_per_epoch):
                img_batch, label_batch, w = sess.run(next_batch)
                print('{} All number: {} Step number: {}'.format(datetime.now(), val_batches_per_epoch, step + 1))
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
            elapsed = (time.clock() - start)
            print(elapsed)
            ## 写入文件
            with open(val_file) as f:
                lines = f.readlines()
            for iline in range(len(lines)):
                # lines[iline] = lines[iline].strip() +  ' ' + str(predictions[iline, 1]) + '\n'
                lines[iline] = lines[iline].strip().split(' ')[1] + ',' + str(predictions[iline, 1]) + '\n'
            print(lines)
            with open(location_file_result, 'w') as f:
                f.write(''.join(lines))
            ############################################################################################################
            c.send(location_file_result.encode("utf-8"))
            ##
            c.close()  # 关闭连接

print('***********************************************************************************************')
print('Done')