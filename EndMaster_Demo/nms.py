#!/usr/bin/env python  
#-*- coding: utf-8 -*-

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def NMS(rectangles,threshold,type):
    if len(rectangles)==0:
	return rectangles
    boxes = np.array(rectangles, dtype=np.float32)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s  = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
	xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) #I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
	w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
	if type == 'iom':
	    o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
	else:
	    o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
	pick.append(I[-1])
	I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick]
    return result_rectangle