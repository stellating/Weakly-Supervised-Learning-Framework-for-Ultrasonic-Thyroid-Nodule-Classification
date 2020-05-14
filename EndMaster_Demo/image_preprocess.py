#!/usr/bin/env python  
#-*- coding: utf-8 -*-

import cv2

## 计算相应方向连续黑色像素的最大长度
def countMaxLength(im_gray, w, h, value_black, row=-1, col=-1):
    ##
    if row >= 0 and col >= 0:
        print 'row >= 0 and col >= 0 !!!'
        exit(0)
    if row < 0 and col < 0:
        print 'row < 0 and col < 0 !!!'
        exit(0)
    ##
    maxLen = 0
    ##
    # print('w, h, row, col = ', w, h, row, col)
    ##
    if row >= 0:
        len_tmp = 0
        for j in range(0, w):
            if im_gray[row, j] <= value_black:
                len_tmp += 1
            else:
                len_tmp = 0
            maxLen = max(maxLen, len_tmp)
    ##
    if col >= 0:
        len_tmp = 0
        for i in range(0, h):
            if im_gray[i, col] <= value_black:
                len_tmp += 1
            else:
                len_tmp = 0
            maxLen = max(maxLen, len_tmp)
    ##
    return maxLen

## 去除周围的无效边框
def removeAround(im_gray, value_black, f_scale_x=0.3, f_scale_y=0.3):
    im_gray = cv2.resize(im_gray, (0, 0), fx=f_scale_x, fy=f_scale_y)
    (h, w) = im_gray.shape
    center_i = h / 2
    center_j = w / 2
    print center_i, center_j
    x1 = center_i; x2 = center_i; y1 = center_j; y2 = center_j
    ##
    prob_w = 1.0 / 2
    prob_h = 1.0 / 2

    min_w = w * prob_w
    min_h = h * prob_h
    ## up
    for i in range(center_i, -1, -1):
        maxLen = countMaxLength(im_gray, w, h, value_black, row=i, col=-1)
        if maxLen < min_w:
            x1 = i
        else:
            break
    ## down
    for i in range(center_i, h):
        maxLen = countMaxLength(im_gray, w, h, value_black, row=i, col=-1)
        if maxLen < min_w:
            x2 = i
        else:
            break
    ## left
    for j in range(center_j, -1, -1):
        maxLen = countMaxLength(im_gray, w, h, value_black, row=-1, col=j)
        if maxLen < min_h:
            y1 = j
        else:
            break
    ## right
    for j in range(center_j, w):
        maxLen = countMaxLength(im_gray, w, h, value_black, row=-1, col=j)
        if maxLen < min_h:
            y2 = j
        else:
            break
    ##
    ##
    x1 = int(x1 / f_scale_x)
    x2 = int(x2 / f_scale_x)
    y1 = int(y1 / f_scale_y)
    y2 = int(y2 / f_scale_y)
    return x1, x2, y1, y2

if __name__ == '__main__':
    print 'test'
    srcfile = './static/input.jpg.bmp'

    im_gray = cv2.imread(srcfile, 0)
    print im_gray.shape

    cv2.imshow('test', im_gray)
    cv2.waitKey(0)
    x1, x2, y1, y2 = removeAround(im_gray, 5, f_scale_x=0.2, f_scale_y=0.2)
    im_crop = im_gray[x1:x2+1, y1:y2+1]
    cv2.imshow('test', im_crop)
    cv2.waitKey(0)