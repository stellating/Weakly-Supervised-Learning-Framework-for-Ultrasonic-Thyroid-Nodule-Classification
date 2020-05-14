#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
直接从location中读取roi
'''

import numpy as np
from keras.models import load_model
import os, datetime
import keras.backend as K
from keras.preprocessing import image

img_width, img_height = 75, 75

batch_size = 256
roiTypeName = '-edge'
imgSuffix_final = '.bmp'
imgPath_origion = 'I:/thyroid-all-gray-20171213/experiments'
testDirPre = 'open_thyroid_'

print('程序开始运行 ...')

t0 = datetime.datetime.now()
K.clear_session()
model = load_model('my_model_0_' + str(img_width) + '_best_1vs5_weight_1vs5_512.h5')

## 该函数接受一个目录，预测下面所有图片的情况，并把结果写入txt文件
## 目录下面有一个dic.txt，指明了每个roi的位置和类别，
def predictDirectDir(imgPath, proposalPath, predictionPath):

    dicData = np.loadtxt(proposalPath, delimiter=',', dtype=float, skiprows=1)
    imOrigion = image.load_img(imgPath)
    print(imOrigion.size)
    ## 将位置变为img_width对应的位置
    # imOrigion.save('F:/thyroid_data/keras/nodule_test/bad_1/nodule/1.bmp');

    xList = [];
    [mr, nr] = dicData.shape;

    for irow in range(0, mr):
        img = imOrigion.crop((int(dicData[irow, 1]) - 1, int(dicData[irow, 2]) - 1, int(dicData[irow, 3]), int(dicData[irow, 4])))
        img = img.resize((img_width, img_height))

        x = image.img_to_array(img) / 255.0;
        xList.append(x)

    xList = np.array(xList)
    print('xList.len = ' + str(len(xList)))
    classes = model.predict(xList, batch_size = batch_size)
    classes = np.array(classes)

    dicData = np.hstack((dicData, classes))

    np.savetxt(predictionPath, dicData, fmt='%.4f, %d, %d, %d, %d, %f\r\n')

    # exit()

for imgType in ['good', 'bad']:
    rootDir_test = imgPath_origion + '/' + testDirPre + imgType
    print('rootDir = ' + rootDir_test)

    imgDir_path = rootDir_test + '/' + 'imgs'
    proposalDir_path = rootDir_test + '/' + 'locations_edge'
    predictDir_path = rootDir_test + '/' + 'prediction_edge'
    if not os.path.exists(predictDir_path):
        os.makedirs(predictDir_path)
    ####################################################################################################################
    filenames = os.listdir(imgDir_path)
    dirCount = len(filenames)
    for iDir in range(0, dirCount):
        imgName = filenames[iDir]
        imgNameOnly = imgName.split('.')[0]
        imgPath = os.path.join(imgDir_path, imgName)
        proposalPath = os.path.join(proposalDir_path, imgNameOnly + '.txt')
        predictionPath = os.path.join(predictDir_path, imgNameOnly + '.txt')
        t1 = datetime.datetime.now()
        print('imgType = ' + imgType + ', imgCount = ' + str(dirCount) + ', iImg = ' + str(iDir)
              + ', imgName = ' + imgName + ', time = ' + str((t1 - t0).seconds) + ' s(seconds)')

        predictDirectDir(imgPath, proposalPath, predictionPath)

        ##
        # break
    ##
    # break
K.clear_session()

print('Done')
t1 = datetime.datetime.now()
print('total time = ' + str((t1 - t0).seconds) + ' s(seconds)')