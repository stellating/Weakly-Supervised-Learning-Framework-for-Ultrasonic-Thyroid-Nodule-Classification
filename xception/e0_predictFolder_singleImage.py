#!/usr/bin/env python  
#-*- coding: utf-8 -*-

import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image
from PIL import ImageEnhance

# bottleneck_features_train = np.ones((3,4))
# print(bottleneck_features_train)
# np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

# train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
# print(train_data.shape[1:])

img_width, img_height = 75, 75
model = load_model('my_model_0_' + str(img_width) + '_best_1vs5_weight_1vs5_512.h5')

nTrain_nodule = 64000;
nTrain_nonnodule = nTrain_nodule * 5;

rootDir_validation = 'I:/thyroid-data-all-gray-20171224/keras/data-' + str(nTrain_nodule) \
                     + '-' + str(nTrain_nonnodule) + '-' + str(img_width) + '/validation/nodule'
thresh = 0.1;
n0 = 0;
for parent, dirnames, filenames in os.walk(rootDir_validation):
    imgCount = len(filenames);
    for iImg in range(0, imgCount):
        filename = filenames[iImg];
        # if iImg % 100 == 0:
            # print('imgCount = ' + str(imgCount) + ', iImg = ' + str(iImg) + ', imgName = ' + filename)
        imgPath = os.path.join(parent, filename)
        # imgPath = u'F:/甲状腺数据/AllTogether/传统多示例训练集/bad-200/bad_30.jpg'
        img = image.load_img(imgPath, target_size=(img_width, img_height))
        # img = image.load_img(imgPath)
        # img.show()
        # print(img)
        contrast = ImageEnhance.Contrast(img)
        img = contrast.enhance(1.0)
        # img.show()
        # print(img)
        # print(contrast)
        # exit(0)

        x = image.img_to_array(img) / 255;
        x = np.expand_dims(x, axis=0)

        classes = model.predict(x)
        yyy = 0;
        if rootDir_validation.find('nonnodule') >= 0:
            yyy = 1;
        if (float(classes[0][0]) < thresh and yyy == 0)\
                or (float(classes[0][0]) >= thresh and yyy == 1):
            n0 += 1
        else:
            print('!!!imgCount = ' + str(imgCount) + ', iImg = ' + str(iImg) + ', imgName = ' + filename)
            print(classes)
            # print(yyy)

    print('nRight = ' + str(n0))
    print('nWrong = ' + str(imgCount - n0))
    print('rightRate = ' + str(n0 / imgCount))