#!/usr/bin/env python  
#-*- coding: utf-8 -*-

'''
本脚本用于对预测的bad_bag的预测值做一个统计，为后面自适应的方法选择参数提供一个参考

自适应的方法仍然从初始化全连接层的mil开始训练
'''

import re
import numpy as np

### 训练了全连接层以后的预测数据
pred_path_1 = 'I:\\thyroid-all-gray-20180124\\trainData\\nodules\\bad_nodules-.txt'

### 按照EM算法跑出来的预测数据
pred_path_2 = 'I:\\thyroid-all-gray-20180124\\trainData\\nodules\\train_process\\finetune_alexnet-20180130\\model-1.0-10.0-0.5\\bad_nodules_pred_mil.txt'

###
def stat_info(pred_path):
    with open(pred_path, 'r') as f:
        lines = f.readlines()
    #
    preds = []
    for line in lines:
        line = line.strip()
        strs = re.split('\\s+', line)
        preds.append(np.float32(strs[2]))
    #
    preds.sort(reverse=True)
    print(len(preds))
    print(preds[3400])
##
##
###############################################################
###############################################################
##
print(pred_path_1)
stat_info(pred_path_1)
##
print(pred_path_2)
stat_info(pred_path_2)
##
##