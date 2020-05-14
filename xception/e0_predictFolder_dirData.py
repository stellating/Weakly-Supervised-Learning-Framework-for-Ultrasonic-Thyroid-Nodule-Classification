#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
直接从location中读取roi
'''

import numpy as np
from keras.models import load_model
import os, datetime
from keras.preprocessing import image
import keras.backend as K
from PIL import Image
import shutil

img_width, img_height = 75, 75
K.clear_session()
model = load_model('my_model_0_75_best_1vs5_weight_1vs5_512.h5')

print('程序开始运行 ...')

batch_size = 512
roiTypeName = '-edge'
imgSuffix_final = '.bmp'

imgPath_origion = 'I:/thyroid-all-gray-20171213/validationData'

tagCheck = True

t0 = datetime.datetime.now()
## 该函数接受一个目录，预测下面所有图片的情况，并把结果写入txt文件
## 目录下面有一个dic.txt，指明了每个roi的位置和类别，
def predictDirectDir(dirPath, imgPathOrigion):
    dicTxtPath = dirPath + '/' + 'dic.txt';

    dicData = np.loadtxt(dicTxtPath, delimiter=',', dtype=float, skiprows=1)
    print('imgPathOrigion = ' + imgPathOrigion)
    imOrigion = image.load_img(imgPathOrigion, grayscale=True)
    print(imOrigion.size)

    xList = [];
    [mr, nr] = dicData.shape;

    for irow in range(0, mr):
        img = imOrigion.crop(
            (int(dicData[irow, 1]) - 1, int(dicData[irow, 2]) - 1, int(dicData[irow, 3]), int(dicData[irow, 4])))
        img = img.resize((img_width, img_height))

        x = image.img_to_array(img) / 255.0;
        xList.append(x)

    xList = np.array(xList)
    print('xList.len = ' + str(len(xList)))
    classes = model.predict(xList, batch_size=batch_size)
    classes = np.array(classes)

    dicData = np.hstack((dicData, classes))

    np.savetxt(dirPath + '/' + '-dic_result.txt', dicData, fmt='%.4f, %d, %d, %d, %d, %.4f\r\n')

### 第一步，先检测是否存在文件夹的数据

if tagCheck:
    for imgType in ['good', 'bad']:
        rootDir_test = imgPath_origion + '/' + imgType + '_prediction'
        if 1 < 2 or os.path.exists(rootDir_test):
            if not os.path.exists(rootDir_test):
                os.mkdir(rootDir_test)
            # 从location文件夹找到相应的文件，并导入看看有没有问题

            locationDir = imgPath_origion + '/' + 'locations_' + imgType + roiTypeName
            for parent, dirnames, filenames in os.walk(locationDir):
                fileCount = len(filenames)
                for iFile in range(0, fileCount):
                    fileName = filenames[iFile]
                    imgNameOnly = fileName.split('.')[0]
                    print('imgType = ' + imgType + ', imgCount = ' + str(fileCount)
                          + ', iImg = ' + str(iFile) + ', imgName = ' + imgNameOnly)
                    locationPath = locationDir + '/' + imgNameOnly + '.txt'
                    resultDir = rootDir_test + '/' + imgNameOnly
                    if not os.path.exists(resultDir):
                        ## 下面这个是测试文件有没有问题，貌似matlab有时候把数组写入文件会出错，导致numpy导入遇到bug
                        dicData = np.loadtxt(locationPath, delimiter=',', dtype=float, skiprows=1)
                        os.mkdir(resultDir)
                        shutil.copy(locationPath, resultDir + '/' + 'dic.txt')

        else:
            print(rootDir_test + ' already exist!!!')
            exit(0)

# exit()
### 开始预测
tag = False
for imgType in ['good', 'bad']:
    #
    # if imgType == 'good':
    #     continue

    rootDir_test = imgPath_origion + '/' + imgType + '_prediction';

    print('rootDir = ' + rootDir_test)
    for parent, dirnames, filenames in os.walk(rootDir_test):
        dirCount = len(dirnames)
        for iDir in range(0, dirCount):
            dirname = dirnames[iDir]

            # if ( not dirname == '867') and (not tag):
            #     continue
            # tag = True

            dirPath = os.path.join(parent, dirname)
            t1 = datetime.datetime.now()
            print('dirCount = ' + str(dirCount) + ', iDir = ' + str(iDir) + ', dirName = ' + dirname + ', time = ' + str((t1 - t0).seconds) + ' s(seconds)')

            imgPath = imgPath_origion + '/' + imgType + '_crop' + '/' + dirname + imgSuffix_final

            predictDirectDir(dirPath, imgPath)
        ##
        break;
K.clear_session()
print('Done')
t1 = datetime.datetime.now()
print('total time = ' + str((t1 - t0).seconds) + ' s(seconds)')