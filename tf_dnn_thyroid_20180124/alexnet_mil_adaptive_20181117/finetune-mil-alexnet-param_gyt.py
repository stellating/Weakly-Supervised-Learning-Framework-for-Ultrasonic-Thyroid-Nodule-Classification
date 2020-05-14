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
#from tensorflow.data import Iterator
from tensorflow.contrib import slim as slim
import alexnet_mil_adaptive_20181117.utils_gyt as utils

########################################################################################################################
### 参数
Cs = [1] ### 表示总共训练次数
Ts = [50] ### 表示每次训练多少epoch
TTT_LAST = 50

count_good = 2706 ### is for data augmentation
# Counts = [[1, 1], [2, 2], [3, 3], [2, 1], [3, 1], [3, 2]] ## 表示数量下降的最高值和最低值
Counts = [[2, 1]] ## 表示数量下降的最高值和最低值
##
weightPositive = 1
weightNegative = 1
thresh_easy = 0.9
##
model_save_dir = 'I:/gyt-data/model/finetune_alexnet_bishe_20200313' \
                 + '_wp_' + str(weightPositive) + '_te_' + str(thresh_easy)
##
if not os.path.exists(model_save_dir):
    os.mkdir(model_save_dir)
else:
    print(model_save_dir, ' already exist!!!')
    ## exit()
########################################################################################################################
########################################################################################################################
########################################################################################################################
### 配置：数据、网络结构、训练参数
########################################################################################################################
###
'''
### 数据

### 真正用于训练网络的数据，每次训练完成后会重新更新一波
train_file = 'I:/thyroid-all-gray-20180124/trainData/nodules/train_process/train.txt'  #6834
### 验证集
val_file_good = 'I:/thyroid-all-gray-20180124/validationData/nodules/validationData-good.txt'  #200
val_file_bad = 'I:/thyroid-all-gray-20180124/validationData/nodules/validationData-bad.txt'   #200
############################################
### 正包示例所组成的数据，每个结节一张图
val_file_bad_bag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules.txt' #7374
### 对上面的文件的预测结果
badPath_predicted = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules_pred_mil.txt'  #7374
### 所有良性数据，已经经过增强了的
goodPath = 'I:/thyroid-all-gray-20180124/trainData/nodules/good_nodules.txt'    #5334
### 确定属于一个包的图片
badDicPath = 'I:/thyroid-all-gray-20180124/trainData/dic_bad.txt'   #7330
badPath = val_file_bad_bag
### 所有恶性结节，已经经过增强了的
badDicPath_allBag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_bag_nodules.txt'   #22122
############################################
'''
### gyt的数据
### 真正用于训练网络的数据，每次训练完成后会重新更新一波
train_file = 'I:/gyt-data/train.txt'  #6834
### 验证集
val_file_good = 'I:/gyt-data/validationData-good.txt'  #200
val_file_bad = 'I:/gyt-data/validationData-bad.txt'   #200
############################################
### 正包示例所组成的数据，每个结节一张图
val_file_bad_bag = 'I:/gyt-data/bad_nodules.txt' #7374
### 对上面的文件的预测结果
badPath_predicted = 'I:/gyt-data/bad_nodules_pred_mil.txt'  #7374
### 所有良性数据，已经经过增强了的
goodPath = 'I:/gyt-data/good_nodules.txt'    #5334
badPath = val_file_bad_bag
badDicPath = 'I:/gyt-data/dic_bad.txt'   #7330
### 所有恶性结节，已经经过增强了的
badDicPath_allBag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_bag_nodules.txt'   #22122
############################################


# Path for tf.summary.FileWriter and to store model checkpoints
#checkpoint_path_fc = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath-20181116/finetune_alexnet_fc_5.0/checkpoints'
checkpoint_path_fc = 'I:/gyt-data/checkpoints'
### 训练参数
propP = 0.0 ## 每个bag选择预测值最高的那一个
thresh_negative = 0.4
###
momentum = 0.9
base_lr = 0.0001 # 最开始的学习率
lr_decay_rate = 0.5
lr_decay_epoch = 1 # 没训练这么多step就重新设置学习率：base_lr *= lr_dacay_rate

batch_size = 16 # 训练的batch_size
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
with tf.device('/gpu:0'):
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
    

'''
# Ops for initializing the two different iterators

validation_good_init_op = iterator.make_initializer(val_data_good.data)
validation_bad_init_op = iterator.make_initializer(val_data_bad.data)
validation_bad_bag_init_op = iterator.make_initializer(val_data_bad_bag.data)
'''
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

    #max_count = 1200; min_count = 1300;

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
                ### 如果是初次训练模型，则从全连接网络加载模型
                if iccc == 0:
                    checkpoint_path_model_last = checkpoint_path_fc
                else: ## 不是初次，则从上一次加载模型
                    checkpoint_path_model_last = os.path.join(model_save_dir_ccc, 'iccc-' + str(iccc - 1), 'checkpoints')
                ##
                model_save_dir_iccc = os.path.join(model_save_dir_ccc, 'iccc-' + str(iccc))
                if not os.path.exists(model_save_dir_iccc):
                    os.mkdir(model_save_dir_iccc)
                #
                filewriter_path = os.path.join(model_save_dir_iccc, 'tensorboard')
                checkpoint_path = os.path.join(model_save_dir_iccc, 'checkpoints')
                # Create parent path if it doesn't exist
                if not os.path.isdir(checkpoint_path):
                    os.makedirs(checkpoint_path)
                else:
                    print(checkpoint_path + ' already exist!!!')
                    # exit(0)
                if not os.path.isdir(filewriter_path):
                    os.makedirs(filewriter_path)
                else:
                    print(filewriter_path + ' already exist!!!')
                    # exit(0)
                ###
                all_counts = np.linspace(max_count, min_count, num=ttt, endpoint=True)  #创建等差数列
                ########################################################################################################
                ## *****************************************************************************************************

                ### 计算step
                train_batches_per_epoch = training_imgs_good_plus_bag // batch_size
                val_good_batches_per_epoch = val_data_good.data_size // batch_size_validation
                val_bad_batches_per_epoch = val_data_bad.data_size // batch_size_validation
                val_bad_bag_batches_per_epoch = np.ceil(val_data_bad_bag.data_size / batch_size_validation).astype(
                    np.int32)

                # List of trainable variables of the layers we want to train
                var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in train_layers]

                # Op for calculating the loss
                with tf.name_scope("cross_ent"):
                    if num_classes > 2 or use_softmax_cross_entropy_loss:
                        print('lossType: softmax_cross_entropy_loss')
                        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,
                        #                                                          labels=y))
                        loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=score, weights=y_weight,
                                                                              onehot_labels=y))
                    elif num_classes == 2:
                        print('lossType: sigmoid_cross_entropy_loss')
                        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score,
                                                                                      labels=y))
                    else:
                        print('wrong num_classes!!!')
                        exit()
                ##
                ##
                with tf.name_scope('my'):
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    learning_rate = tf.train.natural_exp_decay(base_lr, global_step,
                                                           lr_decay_epoch * train_batches_per_epoch, lr_decay_rate,
                                                           staircase=True)
                ##
                # Train op
                with tf.name_scope("train"):
                    # Get gradients of all trainable variables
                    gradients = tf.gradients(loss, var_list)
                    gradients = list(zip(gradients, var_list))

                    # Create optimizer and apply gradient descent to the trainable variables
                    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
                    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum)
                    train_op = optimizer.apply_gradients(grads_and_vars=gradients, global_step=global_step)
                ##
                ### 添加到summary
                for gradient, var in gradients:
                    tf.summary.histogram(var.name + '/gradient', gradient)

                ## 准确率
                with tf.name_scope("accuracy"):
                    correct_pred = tf.equal(tf.argmax(score, 1), tf.argmax(y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
                    p_pred = tf.nn.softmax(score)
                ##
                ### 添加到summary
                tf.summary.scalar('train_accuracy', accuracy)
                tf.summary.scalar('train_loss', loss)
                tf.summary.scalar('learning_rate', learning_rate)
                ##
                # Merge all summaries together
                merged_summary = tf.summary.merge_all()
                writer = tf.summary.FileWriter(filewriter_path)
                ##
                ########################################################################################################################
                ### 开始session过程
                ########################################################################################################################
                config = tf.ConfigProto(allow_soft_placement=True)
                with tf.Session(config=config) as sess:
                    # Initialize all variables
                    sess.run(tf.global_variables_initializer())
                    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                       tr_data.data.output_shapes)
                    
                    validation_good_init_op = iterator.make_initializer(val_data_good.data)
                    validation_bad_init_op = iterator.make_initializer(val_data_bad.data)
                    validation_bad_bag_init_op = iterator.make_initializer(val_data_bad_bag.data)
                    sess.run(validation_good_init_op)
                    sess.run(validation_bad_init_op)
                    sess.run(validation_bad_bag_init_op)

                    next_batch = iterator.get_next()
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
                    print(checkpoint_path_model_last)
                    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path_model_last))
                    ##########

                    print("{} Start training...".format(datetime.now()))
                    print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
                                                                      filewriter_path))
                    ####################################################################################################################
                    ################################################################################################################
                    ### 在验证集上面验证 -- good，先测试一下，可以检查上一次训练的模型与这个模型是不是一样的
                    print("{} Start validation to check".format(datetime.now()))
                    sess.run(validation_good_init_op)
                    w = np.ones(batch_size_validation, dtype=np.float32)
                    test_acc = 0.;
                    test_count = 0;
                    test_loss = 0.
                    for _ in range(val_good_batches_per_epoch):
                        img_batch, label_batch = sess.run(next_batch)[0:2]
                        [acc, loss_tmp] = sess.run([accuracy, loss],
                                                   feed_dict={x: img_batch, y_weight: w, y: label_batch,
                                                              keep_prob: 1.})
                        test_acc += acc;
                        test_loss += loss_tmp;
                        test_count += 1
                    test_acc /= test_count;
                    test_loss /= test_count
                    print("{} Validation Good Accuracy = {:.4f}, Loss = {:.4f}".format(datetime.now(),
                                                                                       test_acc, test_loss))
                    ### 在验证集上面验证 -- bad
                    print("{} Start validation".format(datetime.now()))
                    sess.run(validation_bad_init_op)
                    w = np.ones(batch_size_validation, dtype=np.float32)
                    test_acc = 0.;
                    test_count = 0;
                    test_loss = 0.
                    for _ in range(val_bad_batches_per_epoch):
                        img_batch, label_batch = sess.run(next_batch)[0:2]
                        [acc, loss_tmp] = sess.run([accuracy, loss],
                                                   feed_dict={x: img_batch, y_weight: w, y: label_batch,
                                                              keep_prob: 1.})
                        test_acc += acc;
                        test_loss += loss_tmp;
                        test_count += 1
                    test_acc /= test_count;
                    test_loss /= test_count
                    print("{} Validation Bad Accuracy = {:.4f}, Loss = {:.4f}".format(datetime.now(),
                                                                                      test_acc, test_loss))
                    ####################################################################################################################
                    ### 开始训练
                    ####################################################################################################################
                    for epoch in range(ttt):
                        ##
                        count_bad_imgs = np.int32(all_counts[epoch])
                        ################################################################################################################
                        ################################################################################################################
                        ### 首先选择数据
                        ################################################################################################################
                        ################################################################################################################
                        ################################################################################################################
                        ### 下面重新预测bad bag并修改以进行下一轮迭代
                        print('{} Start validation bad bag'.format(datetime.now()))
                        sess.run(validation_bad_bag_init_op)
                        w = np.ones(batch_size_validation, dtype=np.float32)
                        ####################################################################################################################
                        ### 开始预测
                        ####################################################################################################################
                        test_acc = 0.;
                        test_count = 0;
                        test_loss = 0.;
                        predictions = []
                        for step in range(val_bad_bag_batches_per_epoch):
                            # print('{} All number: {} Step number: {}'.format(datetime.now(),val_bad_bag_batches_per_epoch, step + 1))
                            img_batch, label_batch = sess.run(next_batch)[0:2]
                            if label_batch.shape[0] < batch_size:
                                # print('batch not enough, padding')
                                # print('label.shape = ' + str(label_batch.shape))
                                img_batch_new = np.zeros((batch_size, img_width, img_height, num_channels))
                                label_batch_new = np.zeros((batch_size, 2))
                                img_batch_new[:img_batch.shape[0], :] = img_batch
                                label_batch_new[:label_batch.shape[0], :] = label_batch
                                img_batch = img_batch_new
                                label_batch = label_batch_new
                                # print('img.shape = ' + str(img_batch.shape))
                            # print('label.shape = ' + str(label_batch.shape))

                            [p_value] = sess.run([p_pred],
                                                 feed_dict={x: img_batch, y_weight: w, y: label_batch, keep_prob: 1.0})
                            # print(p_value)
                            predictions.append(p_value)
                        ##
                        ################################################################################################################
                        ### 下面写入文件
                        predictions = np.vstack(predictions)
                        ##
                        utils.writePrediction(badPath_predicted, badPath, predictions)
                        nResult = utils.getNextTrainingTxt(train_file, goodPath, badDicPath, badPath_predicted,
                                                           badDicPath_allBag,
                                                           prop=propP, minCount=1, thresh_negative=thresh_negative,
                                                           count_need=count_bad_imgs, thresh_easy=thresh_easy)
                        ##
                        print('nResult = ' + str(nResult))
                        tr_data.read_txt_file()
                        tr_data.resetData()
                        ##
                        # exit(0)
                        ################################################################################################################
                        ################################################################################################################
                        print()
                        print(
                            '******************************************************************************************')
                        print("{} Epoch number: {}".format(datetime.now(), epoch + 1))
                        print('learning_rate = ', sess.run(learning_rate))
                        print('global_step = ', sess.run(global_step))
                        ### Initialize iterator with the training dataset
                        #training_init_op = iterator.make_initializer(tr_data.data)
                        #sess.run(training_init_op)
                        ## 根据新的这一批数据更新batch count
                        train_batches_per_epoch = nResult // batch_size
                        ##
                        print('train_batches_per_epoch = ', train_batches_per_epoch)
                        print('all_training_sample.count = ', train_batches_per_epoch * batch_size)
                        print('tr_data.data_size = ', tr_data.data_size)
                        ##
                        for step in range(train_batches_per_epoch):
                            training_init_op = iterator.make_initializer(tr_data.data)
                            sess.run(training_init_op)
                            # print('\tstep = ', step, ', train_batches_per_epoch = ', train_batches_per_epoch)
                            img_batch, label_batch = sess.run(next_batch)[0:2]
                            ##
                            if not img_batch.shape[0] == batch_size or not label_batch.shape[0] == batch_size:
                                print('img_batch.shape[0] = ', img_batch.shape[0])
                                print('label_batch.shape[0] = ', label_batch.shape[0])
                                continue
                                # exit(0)
                            ##################################################
                            w = np.ones(img_batch.shape[0], dtype=np.float32)
                            for ilabel in range(label_batch.shape[0]):
                                if label_batch[ilabel][0] > 0.5:  ## 属于第0类
                                    w[ilabel] = weight_class[0]
                                else:  ## 属于第1类
                                    w[ilabel] = weight_class[1]
                            ##################################################
                            sess.run(train_op,
                                     feed_dict={x: img_batch, y_weight: w, y: label_batch, keep_prob: dropout_rate})
                            ### 将summary信息保存
                            if step % display_step == 0:
                                s = sess.run(merged_summary,
                                             feed_dict={x: img_batch, y_weight: w, y: label_batch, keep_prob: 1.})
                                writer.add_summary(s, epoch * train_batches_per_epoch + step)
                            ##
                            #print('\t\tstep end')
                            #break
                        ################################################################################################################
                        ################################################################################################################
                        ### 在验证集上面验证 -- good
                        print("{} Start validation".format(datetime.now()))
                        sess.run(validation_good_init_op)
                        w = np.ones(batch_size_validation, dtype=np.float32)
                        test_acc = 0.;
                        test_count = 0;
                        test_loss = 0.
                        for _ in range(val_good_batches_per_epoch):
                            img_batch, label_batch = sess.run(next_batch)[0:2]
                            [acc, loss_tmp] = sess.run([accuracy, loss],
                                                       feed_dict={x: img_batch, y_weight: w, y: label_batch,
                                                                  keep_prob: 1.})
                            test_acc += acc;
                            test_loss += loss_tmp;
                            test_count += 1
                        test_acc /= test_count;
                        test_loss /= test_count
                        print("{} Validation Good Accuracy = {:.4f}, Loss = {:.4f}".format(datetime.now(),
                                                                                           test_acc, test_loss))
                        ### 在验证集上面验证 -- bad
                        print("{} Start validation".format(datetime.now()))
                        sess.run(validation_bad_init_op)
                        w = np.ones(batch_size_validation, dtype=np.float32)
                        test_acc = 0.;
                        test_count = 0;
                        test_loss = 0.
                        for _ in range(val_bad_batches_per_epoch):
                            img_batch, label_batch = sess.run(next_batch)[0:2]
                            [acc, loss_tmp] = sess.run([accuracy, loss],
                                                       feed_dict={x: img_batch, y_weight: w, y: label_batch,
                                                                  keep_prob: 1.})
                            test_acc += acc;
                            test_loss += loss_tmp;
                            test_count += 1
                        test_acc /= test_count;
                        test_loss /= test_count
                        print("{} Validation Bad Accuracy = {:.4f}, Loss = {:.4f}".format(datetime.now(),
                                                                                          test_acc, test_loss))
                        ################################################################################################################
                        ################################################################################################################
                        ################################################################################################################
                        print("{} Saving checkpoint of model...".format(datetime.now()))
                        ### 保存checkpoint
                        checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
                        save_path = saver.save(sess, checkpoint_name, global_step=global_step)
                        print('save_path = ' + save_path)
                        print("{} Model checkpoint saved at {}".format(datetime.now(), checkpoint_name))
                        ####################################################################################################################
                        ## end for : epoch
                        ##

                ########################################################################################################

                ###
            ## end ttt
        ## end 每一次
    ### end ccc
## end for cccount

print('Done')
print('res_path = ', model_save_dir)
