#!/usr/bin/env python  
#-*- coding: utf-8 -*-

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft

# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
import os
import nms as nmsorigin
import nms_modified as nms_modified
import cv2

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
            #print('nms_modified: 加权平均')
            # boxes_together = np.array(((1,2,3,4, 0.3),(5,6,7,8, 0.4)))
            s = boxes_together[:, -1]
            s = np.reshape(s, (-1, 1))
            s_a = np.sum(s)
            boxes_together[:, :] *= s
            box = np.sum(boxes_together, axis=0)
            box /= s_a
        else:
            #print('nms_modified: 平均')
            box = np.mean(boxes_together, axis = 0)
        #
        res_box.append(box)
    return np.array(res_box, dtype=np.float32)

if __name__ == "__main__":
    #print '原始NMS'
    imgdir='D:\\project\\wjx\\preprocessed_data_RGB\\'
    imgsavedir = 'D:\\project\\wjx\\nms_result_visualize\\'
    filedir = 'D:\\project\\wjx\\nms_result\\'
    vgg_res_dir='D:\\project\\wjx\\vgg_result\\'
    vgg_res_list=os.listdir(vgg_res_dir)
    for vgg_respath in vgg_res_list:
        infile=vgg_respath.split('_')[0]
        imgpath=os.path.join(imgdir,infile)
        im=cv2.imread(imgpath)
        vgg_res_path=os.path.join(vgg_res_dir,vgg_respath)
        vgg_res = np.loadtxt(vgg_res_path, delimiter=',', dtype=float)
        print vgg_res.shape
        if vgg_res.shape[0]==0.:
            continue
        if len(vgg_res.shape)==1:
            vgg_res=vgg_res[np.newaxis,:]
        boxes_nms = nmsorigin.NMS(vgg_res, threshold=0.3, type='iou')
        (nboxes, _) = boxes_nms.shape
        im_nms = im.copy()
        for ib in range(nboxes):
            x1 = np.int16(boxes_nms[ib, 0]); y1 = np.int16(boxes_nms[ib, 1]);
            x2 = np.int16(boxes_nms[ib, 2]); y2 = np.int16(boxes_nms[ib, 3]);
            cv2.rectangle(im_nms, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ##
        nmsfile = imgsavedir+infile + '-3-nms' + '.jpg'
        cv2.imwrite(nmsfile, im_nms)
        #print 'nmsfile = ', nmsfile
        #############################################################################
        ## 第六步：
        ## 改进的NMS
        #print '改进的NMS'
        boxes_nms_modified = nms_modified.nms(vgg_res, boxes_nms, thresh=0.4, weighted=True)
        # boxes_nms = vgg_res[boxes_nms, :]
        # boxes_nms_modified = nms.NMS(boxes_nms_modified, threshold=0.3, type='iou')
        #print boxes_nms
        (nboxes, _) = boxes_nms_modified.shape
        im_nms = im.copy()
        for ib in range(nboxes):
            x1 = np.int16(boxes_nms_modified[ib, 0]);
            y1 = np.int16(boxes_nms_modified[ib, 1]);
            x2 = np.int16(boxes_nms_modified[ib, 2]);
            y2 = np.int16(boxes_nms_modified[ib, 3]);
            cv2.rectangle(im_nms, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ##
        nms_modified_img =imgsavedir+infile + '-4-nms-modified' + '.jpg'
        cv2.imwrite(nms_modified_img, im_nms)
        print 'nms_modified_file = ', nms_modified_img
        #
        nms_modified_file = filedir + '/' + infile+'bbox_nms_modified.txt'
        np.savetxt(nms_modified_file, boxes_nms_modified, fmt='%d, %d, %d, %d, %.4f')
        #print os.path.abspath(nms_modified_file)