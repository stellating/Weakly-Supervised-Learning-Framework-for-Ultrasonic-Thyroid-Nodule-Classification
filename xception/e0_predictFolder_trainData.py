#!/usr/bin/env python
#-*- coding: utf-8 -*-

'''
直接从location中读取roi
'''

import numpy as np
from keras.models import load_model
import os, datetime
from keras.preprocessing import image
from PIL import Image

img_width, img_height = 75, 75
model = load_model('my_model_0_' + str(img_width) + '_best_1vs5_weight_1vs5_512.h5')

print('程序开始运行 ...')

batch_size = 512
roiTypeName = '-edge'
imgSuffix_final = '.bmp'

imgPath_origion = 'I:/thyroid-recognition-20171213/trainData/'
imgDirPath = imgPath_origion + 'imgs'
locationDirPath = imgPath_origion + 'iou_result' + roiTypeName
locationDirPath_result = imgPath_origion + 'iou_result' + roiTypeName + '-prediction-1'

if os.path.exists(locationDirPath_result):
    print(locationDirPath_result + ' already exists!!!')
    # exit(0)
else:
    os.mkdir(locationDirPath_result)

## 该函数接受一个目录，预测下面所有图片的情况，并把结果写入txt文件
## 目录下面有一个dic.txt，指明了每个roi的位置和类别，
def predictDirectDir(imgPath, locationPath, locationPath_result):
    t0 = datetime.datetime.now()
    dicData = np.loadtxt(locationPath, delimiter=',', dtype=float, skiprows=1)
    # return
    imOrigion = image.load_img(imgPath)
    print(imOrigion.size)
    print(dicData.shape)

    xList = [];
    [mr, nr] = dicData.shape;
    for irow in range(0, mr):
        img = imOrigion.crop((int(dicData[irow, 1]) - 1, int(dicData[irow, 2]) - 1, int(dicData[irow, 3]), int(dicData[irow, 4])))
        img = img.resize((img_width, img_height))

        x = image.img_to_array(img) / 255.0;
        xList.append(x)

    xList = np.array(xList)
    print('xList.len = ' + str(len(xList)))
    classes = model.predict(xList, batch_size=batch_size)
    classes = np.array(classes)
    dicData = np.hstack((dicData, classes))
    print(dicData[0:2, :])

    np.savetxt(locationPath_result, dicData, fmt='%.4f, %d, %d, %d, %d, %.4f, %.4f\r\n')
    t1 = datetime.datetime.now()
    print('time = ' + str((t1 - t0).seconds) + ' s(seconds)')
    # exit()
## 程序开始运行
t0 = datetime.datetime.now()
tag = False
for parent, dirnames, filenames in os.walk(imgDirPath):
    imgCount = len(filenames)
    for iImg in range(0, imgCount): # 遍历每一张图片
        imgName = filenames[iImg]
        imgNameOnly = imgName.split('.')[0]
        print(imgNameOnly)
        if (not imgNameOnly == 'good_76') and (not tag):
            continue
        tag = True

        imgPath = os.path.join(imgDirPath, imgName)
        locationPath = os.path.join(locationDirPath, imgNameOnly + '.txt')

        if not os.path.exists(locationDirPath_result + '/' + imgNameOnly):
            os.mkdir(locationDirPath_result + '/' + imgNameOnly)

        locationPath_result = locationDirPath_result + '/' + imgNameOnly + '/' + 'prediction_result.txt'

        print('imgPath = ' + imgPath)
        print('locationPath = ' + locationPath)
        predictDirectDir(imgPath, locationPath, locationPath_result)

        # break
    ##
    break

print('Done')
t1 = datetime.datetime.now()
print('total time = ' + str((t1 - t0).seconds) + ' s(seconds)')