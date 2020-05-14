#!/usr/bin/env python  
#-*- coding: utf-8 -*-

import os, shutil, re

imgDirLocal = 'J:\Wangjianxiong_20180123'

caffe_prediction_path = 'J:\Wangjianxiong_20180123\wangjianxiong.txt'
f = open(caffe_prediction_path)
lines = f.readlines()
f.close()

nImgs = 0
nLines = len(lines)

iLine = 0
nNotFound = 0
while iLine < nLines:
    line = lines[iLine].strip()
    if line.find('det') >= 0:
        iLine += 1
        continue

    if line.endswith('))'):
        iLine += 1
        continue

    if line.endswith('Check Caffe OK!'):
        iLine += 1
        continue

    ## 到了文件名
    print(line)
    imgPath = line
    imgNames = line.split('/')

    imgDir = imgNames[4]
    imgName = imgNames[5]
    imgNameOnly = imgName.split('.')[0]
    print(imgDir + '\\' + imgNameOnly)

    ## xmin
    iLine += 1
    line = lines[iLine].strip()
    matchObj = re.match('.*array\((.*), dtype.*$', line)
    if matchObj:
        xmin = eval(matchObj.group(1))
    else:
        print('no match')

    ## xmin
    iLine += 1
    line = lines[iLine].strip()
    matchObj = re.match('.*array\((.*), dtype.*$', line)
    if matchObj:
        ymin = eval(matchObj.group(1))
    else:
        print('no match')

    ## xmin
    iLine += 1
    line = lines[iLine].strip()
    matchObj = re.match('.*array\((.*), dtype.*$', line)
    if matchObj:
        xmax = eval(matchObj.group(1))
    else:
        print('no match')

    ## xmin
    iLine += 1
    line = lines[iLine].strip()
    matchObj = re.match('.*array\((.*), dtype.*$', line)
    if matchObj:
        ymax = eval(matchObj.group(1))
    else:
        print('no match')

    print(xmin)
    print(ymin)
    print(xmax)
    print(ymax)

    strXieru = ''
    ###############
    if len(xmin) == 0:
        nNotFound += 1
    for ip in range(0, len(xmin)):
        x1 = xmin[ip]; y1 = ymin[ip];
        x2 = xmax[ip]; y2 = ymax[ip];
        strXieru += str(x1) + ', ' + str(y1) + ', ' + str(x2) + ', ' + str(y2) + '\n'

    print(strXieru)

    if not os.path.exists(imgDirLocal + '\\' + imgDir + '_location_caffe'):
        os.mkdir(imgDirLocal + '\\' + imgDir + '_location_caffe')

    locationPath = imgDirLocal + '\\' + imgDir + '_location_caffe' + '\\' + imgNameOnly + '.txt'
    print(locationPath)

    f = open(locationPath, 'w')
    f.write(strXieru)
    f.close()
    ###############
    nImgs += 1
    iLine += 1

    ###############
    # if nImgs == 6:
    #     break

print('nImgs = ' + str(nImgs))
print('nNotFound = ' + str(nNotFound))