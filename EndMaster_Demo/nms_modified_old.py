#!/usr/bin/env python  
#-*- coding: utf-8 -*-

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np

def nms(dets, dets_nms, thresh, weighted=False):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    ##
    (nboxes, _) = dets_nms.shape
    res_box = []
    for ib in range(nboxes):

        xx1 = np.maximum(dets_nms[ib, 0], x1)
        yy1 = np.maximum(dets_nms[ib, 1], y1)
        xx2 = np.minimum(dets_nms[ib, 2], x2)
        yy2 = np.minimum(dets_nms[ib, 3], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        areas_this = (dets_nms[ib, 2] - dets_nms[ib, 0] + 1) * (dets_nms[ib, 3] - dets_nms[ib, 1] + 1)
        ovr = inter / (areas_this + areas - inter)

        inds = np.where(ovr >= thresh)[0]
        #
        boxes_together = dets[inds, :]
        #
        if weighted:
            # 加权平均
            print('nms_modified: 加权平均')
            # boxes_together = np.array(((1,2,3,4, 0.3),(5,6,7,8, 0.4)))
            s = boxes_together[:, -1]
            s = np.reshape(s, (-1, 1))
            s_a = np.sum(s)
            boxes_together[:, :] *= s
            box = np.sum(boxes_together, axis=0)
            box /= s_a
        else:
            print('nms_modified: 平均')
            box = np.mean(boxes_together, axis = 0)
        #
        res_box.append(box)
    return np.array(res_box, dtype=np.float32)