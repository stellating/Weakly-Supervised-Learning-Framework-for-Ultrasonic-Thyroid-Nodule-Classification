#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
    本脚本用来从open database中随机选择一些作为训练集，其他的作为测试集
'''
import os, shutil, random

#### 下面这个仅仅是恶性图片的名字
badImgNameDir_path_tmp = 'I:/thyroid-all-gray-20171213/experiments/open_thyroid_bad/imgs'
goodImgNameDir_path_tmp = 'I:/thyroid-all-gray-20180124/openData-doctor/good_crop'
####
badImgNameDir_path = 'I:/thyroid-all-gray-20180124/openData-doctor/bad_crop'
badImgLocationDir_path = 'I:/thyroid-all-gray-20180124/openData-doctor/bad_location'
goodImgNameDir_path = 'I:/thyroid-all-gray-20180124/openData-doctor/good_crop'
goodImgLocationDir_path = 'I:/thyroid-all-gray-20180124/openData-doctor/good_location'
####

nTrain_good = 60 ## 因为良性总共有70张图片，所以取出60张来
nTrain_bad = 80 ## 为了避免类别不均衡，干脆恶性和良性的训练数据一样多，其他的都是测试集
imgSuffix = '.bmp'
imgSuffix_final = '.bmp'
########################################################################################################################
resultDir_path = 'I:/thyroid-all-gray-20180124/openData-finetune-classification-' + str(nTrain_good) + '-' + str(nTrain_bad) + '/'
if os.path.exists(resultDir_path):
    print(resultDir_path + ' already exist !!!')
    exit()
else:
    os.makedirs(resultDir_path)
########
open_train_good_img = resultDir_path + 'train/good_crop'
open_train_good_location = resultDir_path + 'train/good_location'
open_train_bad_img = resultDir_path + 'train/bad_crop'
open_train_bad_location = resultDir_path + 'train/bad_location'
open_validation_good_img = resultDir_path + 'validation/good_crop'
open_validation_good_location = resultDir_path + 'validation/good_location'
open_validation_bad_img = resultDir_path + 'validation/bad_crop'
open_validation_bad_location = resultDir_path + 'validation/bad_location'
########################################################################################################################

def getTrainAndValidation(goodImgNameDir_path_tmp, goodImgNameDir_path, goodImgLocationDir_path,
                      open_train_good_img, open_train_good_location,
                      open_validation_good_img, open_validation_good_location, nTrain_good):

    #######
    if os.path.exists(open_train_good_img):
        print(open_train_good_img + ' already exist !!!')
        # exit()
    else:
        os.makedirs(open_train_good_img)
    if os.path.exists(open_train_good_location):
        print(open_train_good_location + ' already exist !!!')
        # exit()
    else:
        os.makedirs(open_train_good_location)
    if os.path.exists(open_validation_good_img):
        print(open_validation_good_img + ' already exist !!!')
        # exit()
    else:
        os.makedirs(open_validation_good_img)
    if os.path.exists(open_validation_good_location):
        print(open_validation_good_location + ' already exist !!!')
        # exit()
    else:
        os.makedirs(open_validation_good_location)

    #######
    imgNames = os.listdir(goodImgNameDir_path_tmp)
    nImgs = len(imgNames)
    print('nImgs = ' + str(nImgs))

    random.shuffle(imgNames)

    nTrain_tmp = 0
    for iImg in range(0, nImgs):
        imgPath_old = imgNames[iImg]
        print(imgPath_old)
        imgNameOnly = imgPath_old.split('.')[0]

        imgPath = goodImgNameDir_path + '/' + imgNameOnly + imgSuffix
        locationPath = goodImgLocationDir_path + '/' + imgNameOnly + '.txt'
        print(imgPath)
        print(locationPath)

        if nTrain_tmp < nTrain_good:
            shutil.copy(imgPath, open_train_good_img + '/' + imgNameOnly + imgSuffix_final)
            shutil.copy(locationPath, open_train_good_location + '/' + imgNameOnly + '.txt')
            nTrain_tmp += 1
        else:
            shutil.copy(imgPath, open_validation_good_img + '/' + imgNameOnly + imgSuffix_final)
            shutil.copy(locationPath, open_validation_good_location + '/' + imgNameOnly + '.txt')

getTrainAndValidation(goodImgNameDir_path_tmp, goodImgNameDir_path, goodImgLocationDir_path,
                      open_train_good_img, open_train_good_location,
                      open_validation_good_img, open_validation_good_location, nTrain_good)

getTrainAndValidation(badImgNameDir_path_tmp, badImgNameDir_path, badImgLocationDir_path,
                      open_train_bad_img, open_train_bad_location,
                      open_validation_bad_img, open_validation_bad_location, nTrain_bad)