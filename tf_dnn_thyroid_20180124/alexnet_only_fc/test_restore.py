#!/usr/bin/env python  
#-*- coding: utf-8 -*-

import tensorflow as tf

# Path for tf.summary.FileWriter and to store model checkpoints
checkpoint_path = 'I:/thyroid-all-gray-20171213/trainData/ChoseForBenignManign-Alex-227/modelPath/finetune_alexnet_fc/checkpoints'
meta_name = 'model_epoch20.ckpt-3960.meta'

##############################################

sess = tf.Session()
saver = tf.train.import_meta_graph(checkpoint_path + '/' + meta_name)
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

graph = tf.get_default_graph()

############################
# op_to_restore = graph.get_tensor_by_name("op_to_restore:0")
all_tensors = graph.get_all_collection_keys()
for iKey in range(len(all_tensors)):
    print('iKey = ' + str(iKey) + ', key = ' + all_tensors[iKey])

train_op = graph.get_collection('train_op')
print(train_op)

tvs = [v for v in tf.trainable_variables()]
for v in tvs:
    print(v.name)
    print(sess.run(v))
