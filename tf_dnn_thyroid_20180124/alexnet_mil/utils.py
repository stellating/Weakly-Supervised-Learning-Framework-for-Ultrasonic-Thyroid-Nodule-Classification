#!/usr/bin/env python  
#-*- coding: utf-8 -*-

'''
    一些帮助类
'''
import random, re
import numpy as np

########################
### 该函数用来编写下一次用来训练的txt文件
########################
def getNextTrainingTxt(resultPath, goodPath, badDicPath, badPath_predicted, badDicPath_allBag,
                       prop=0.3, minCount=1, thresh=0.5):
    ### 保存在这个list里面，这样可以shuffle一下
    listResult = []

    ### 先写入good文件
    with open(goodPath) as f:
        listResult.extend(f.readlines())

    ### 然后预测bad bag
    trainBagImageId = [] ## 这个用来记录选择的bad图片的id
    ####################################################################################################################

    ### 首先获取所有bag
    dic_idToBagname = {}
    with open(badDicPath) as f:
        lines = f.readlines()
        for iLine in range(0, len(lines)):
            line = lines[iLine].strip()
            linsSplit = line.split('\t')
            imgPath_origin = linsSplit[1]
            imgPath_now = linsSplit[2]

            imgPath_origins = imgPath_origin.split('\\');
            bagNameNow = imgPath_origins[-2].strip()
            imgPath_now = imgPath_now.split('/')[-1];
            imgId = imgPath_now.split('.')[0].strip()

            dic_idToBagname[imgId] = bagNameNow
    ### 遍历一遍预测后的文件，记录每个bag的图像名字以及得分
    dic_bag_imgId = {}; dic_bag_pred = {};
    with open(badPath_predicted) as f:
        lines = f.readlines()
        for iLine in range(0, len(lines)):
            line = lines[iLine].strip()
            linsSplit = line.split(' ')
            imgPath_now = linsSplit[0]
            pred = float(linsSplit[2])

            imgPath_now = imgPath_now.split('/')[-1];

            matchObj = re.match('bad_(\\d+_\\d+)\..*', imgPath_now)
            if matchObj:
                imgId = matchObj.group(1)
                # print('imgId = ' + imgId)
            else:
                print('no match')
                exit()
            imgId_this_img = imgId.split('_')[0]
            bagName = dic_idToBagname[imgId_this_img]
            # print('bagName = ' + bagName)

            if bagName not in dic_bag_imgId:
                dic_bag_imgId[bagName] = [imgId]
                dic_bag_pred[bagName] = [pred]
            else:
                dic_bag_imgId[bagName].append(imgId)
                dic_bag_pred[bagName].append(pred)

    ###
    # print(len(dic_bag_imgId))
    # print(len(dic_bag_pred))

    bagKeys = list(dic_bag_imgId)
    for iBag in range(0, len(bagKeys)):
        bagName = bagKeys[iBag]
        ids = dic_bag_imgId[bagName]
        preds = dic_bag_pred[bagName]
        # print(ids)
        # print(preds)

        preds = np.array(preds)
        index = np.argsort(-preds)

        topK = max(minCount, np.int16(len(ids) * prop)) - 1
        kthValue = preds[index[topK]]

        for ieach in range(0, len(ids)):
            if preds[ieach] >= kthValue:
                trainBagImageId.append(ids[ieach])

    # print(trainBagImageId)
    # exit()

    ####################################################################################################################
    ### 根据trainBadImageId来选择其他图片
    with open(badDicPath_allBag) as f:
        lines = f.readlines()
        for iLine in range(0, len(lines)):
            line = lines[iLine].strip()
            imgPath = line.split(' ')[0]
            imgName = imgPath.split('/')[-1].strip()
            # print('imgName = ' + imgName)
            matchObj = re.match('bad_(.*_.*)_\\d+.*', imgName)
            if matchObj:
                imgId = matchObj.group(1)
                # print('imgId = ' + imgId)
            else:
                print('no match')
                exit()
            if imgId in trainBagImageId:
                listResult.append(imgPath.strip() + ' ' + str(1))

    ### 写入文件
    random.shuffle(listResult)
    with open(resultPath, 'w') as f:
        for line in listResult:
            f.write(line.strip() + '\n')

    return len(listResult)

########################
### 该函数用来将当前bad文件的预测结果写入另一个文件
########################
def writePrediction(resultPath, badPath, predictions):
    f = open(badPath)
    val_lines = f.readlines()
    f.close()
    strXieru = ''
    for iLine in range(0, len(val_lines)):
        val_line = val_lines[iLine].strip()
        strXieru += val_line + ' ' + str(predictions[iLine][1]) + '\n'
    with open(resultPath, 'w') as f:
        f.write(strXieru)

####################################################################
####################################################################
#### 测试
# resultPath = 'I:/thyroid-all-gray-20180124/trainData/nodules/choose_for_train_after_fc.txt'
# goodPath = 'I:/thyroid-all-gray-20180124/trainData/nodules/good_nodules.txt'
# badPath_predicted = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_nodules_pred_fc.txt'
# badDicPath = 'I:/thyroid-all-gray-20180124/trainData/dic_bad.txt'
# badDicPath_allBag = 'I:/thyroid-all-gray-20180124/trainData/nodules/bad_bag_nodules.txt'

# nResult = getNextTrainingTxt(resultPath, goodPath, badDicPath, badPath_predicted,
#                    badDicPath_allBag, prop=0.3, minCount=1, thresh=0.5)
#
# print('nResult = ' +str(nResult))