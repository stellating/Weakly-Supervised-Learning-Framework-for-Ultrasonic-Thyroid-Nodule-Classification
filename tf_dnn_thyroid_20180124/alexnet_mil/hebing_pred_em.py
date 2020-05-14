#!/usr/bin/env python  
#-*- coding: utf-8 -*-

'''
# 合并几个不同的模型的预测结果，用于emsemble
'''

import numpy as np
import os
##
root_dir = 'I:/thyroid-all-gray-20180124/testData/nodules'

prefix = 'testData-em-3-3-'
txtNames = ['0', '1', '2', '3', '4']
resultTxt = prefix + 'all.txt'
##
def loadData(txtPath):
    print('txtPath = ' + txtPath)
    with open(txtPath, 'r') as f:
        lines = f.readlines()
    nLine = len(lines)
    labels = []
    preds = []
    for iline in range(nLine):
        line = lines[iline].strip()
        strs = line.split(',')
        labels.append(strs[0])
        preds.append(np.float32(strs[1]))
    #
    print(labels)
    print(preds)

    return labels, preds
#
predsAll = []
#
for txtName in txtNames:
    labels, preds = loadData(os.path.join(root_dir, prefix + txtName + '.txt'))
    predsAll.append(preds)
#
print('**** done ****')
print(labels)
predsAll = np.array(predsAll, dtype=np.float32)
print(predsAll.shape)
predsFinal = np.mean(predsAll, axis=0)
print(predsFinal.shape)
print(predsFinal)

##
with open(os.path.join(root_dir, resultTxt), 'w') as f:
    nLines = len(labels)
    for iline in range(nLines):
        f.write(labels[iline] + ',' + str(predsFinal[iline]) + '\n')
##
print('**** done ****')