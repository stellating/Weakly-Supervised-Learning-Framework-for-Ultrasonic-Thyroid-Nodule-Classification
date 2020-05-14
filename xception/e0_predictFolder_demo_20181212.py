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
import scipy.io as scio
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##

'''
#import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))
'''
##
img_width, img_height = 75, 75
batch_size = 256
roiTypeName = '-edge'
imgSuffix_final = '.bmp'

K.clear_session()
model = load_model('my_model_0_' + str(img_width) + '_best_1vs5_weight_1vs5_512.h5')

## 该函数接受一个目录，预测下面所有图片的情况，并把结果写入txt文件
## 目录下面有一个dic.txt，指明了每个roi的位置和类别，
def predictDirectDir(dicData, imgPathOrigion, dirPath):

    (n_boxes, cols) = dicData.shape
    if n_boxes == 0:
        np.savetxt(dirPath + '/' + imgPathOrigion.split('\\')[-1]+'_vgg16-result.txt', dicData, fmt='%d, %d, %d, %d, %.4f')
        return dirPath + '/' + imgPathOrigion.split('\\')[-1]+'_vgg16-result.txt'

    imOrigion = image.load_img(imgPathOrigion)
    print(imOrigion.size)
    ## 将位置变为img_width对应的位置
    # imOrigion.save('F:/thyroid_data/keras/nodule_test/bad_1/nodule/1.bmp');

    xList = [];
    [mr, nr] = dicData.shape;
    for irow in range(0, mr):
        img = imOrigion.crop((int(dicData[irow, 0]), int(dicData[irow, 1]), int(dicData[irow, 2] + 1), int(dicData[irow, 3]) + 1))
        img = img.resize((img_width, img_height))

        x = image.img_to_array(img) / 255.0;
        xList.append(x)

    xList = np.array(xList)
    print('xList.len = ' + str(len(xList)))
    classes = model.predict(xList, batch_size = batch_size)
    classes = 1.0 - np.array(classes)

    dicData = np.hstack((dicData, classes))

    dicData = dicData[dicData[:, 4] > 0.9, :]

    np.savetxt(dirPath + '/' + imgPathOrigion.split('\\')[-1]+'_vgg16-result.txt', dicData, fmt='%d, %d, %d, %d, %.4f')

    return 0
'''
##
import socket               # 导入 socket 模块

i_start = 0
while True:
    print()
    print('**********************************');
    print('count = ', i_start + 1)
    i_start += 1
    #
    s = socket.socket()         # 创建 socket 对象
    host = '219.224.167.160' # 获取本地主机名
    port = 9999                # 设置端口
    s.bind((host, port))        # 绑定端口
    s.listen(5)                 # 等待客户端连接
    print('receiving ...')

    c, addr = s.accept()     # 建立客户端连接。
    data = c.recv(2048)
    data = data.decode("utf-8")
    print(data)
    ##
    strs = data.split(';;;')
    imgPath = strs[0]
    bbsPath = strs[1]
    #
    dirPath = os.path.dirname(imgPath)
    #
    bbs = scio.loadmat(bbsPath)
    print(bbs['bbs'].shape)
    vgg_res = predictDirectDir(bbs['bbs'], imgPath, dirPath)
    ##
    c.send(vgg_res.encode("utf-8"))
    ##
    c.close()                # 关闭连接

##
K.clear_session()
'''
def main():
    imgdir = 'D:\\project\\wjx\\preprocessed_data_RGB\\'
    bbsdir = 'D:\\project\\wjx\\matfile\\'
    dirPath = 'D:\\project\\wjx\\vgg_result'
    bbslist = os.listdir(bbsdir)
    for bbs in bbslist:
        if int(bbs.split('.')[0])<614:
            continue
        print(bbs)
        bbsPath = os.path.join(bbsdir,bbs)
        imgname = bbs.split('.')[0]+'.'+bbs.split('.')[1]
        imgPath = os.path.join(imgdir,imgname)
        bbs = scio.loadmat(bbsPath)
        print(bbs['bbs'].shape)
        predictDirectDir(bbs['bbs'], imgPath, dirPath)
    print('Done')

if __name__ == "__main__":
    main()
