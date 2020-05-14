#!/usr/bin/env python  
#-*- coding: utf-8 -*-

import web
import cv2
import os
import image_preprocess as image_preprocess
import scipy.io as scio
import numpy as np
import nms as nms
import nms_modified as nms_modified
import socket, time
##
#web.config.debug = False
##
render = web.template.render('templates/')

##
urls = (
    '/', 'index',
    '/hello', 'index_code',
    '/upload', 'upload'
)

#!======================================================================================================================
##
class index:
    def GET(self):
        return render.index()
##
class index_code:
    def GET(self):
        input = web.input(name=None)
        print input.name
        return render.index_code(input.name)
##
class upload:

    def GET(self):
        web.header("Content-Type","text/html; charset=utf-8")
        return render.upload('', '', '', '', '', '')

    def POST(self):
        x = web.input(myfile={})

        filedir = './static' # change this to the directory you want to store the file in.
        if 'myfile' in x:
            #############################################################################
            ## 第一步：
            ## 保存提交的原图
            print '保存提交的原图'
            filepath = x.myfile.filename.replace('\\','/')
            filename = filepath.split('/')[-1]
            filename = 'input' + '.' + filename.split('.')[-1]

            infile = filedir + '/' + filename
            fout = open(infile, 'wb')
            fout.write(x.myfile.file.read())
            fout.close()
            srcfile = infile + '.bmp'
            im = cv2.imread(infile)
            cv2.imwrite(srcfile, im)
            #############################################################################
            ## 第二步：
            ## 图像预处理，包括rgb转灰度图，以及去除无用边框
            print '图像预处理，包括rgb转灰度图，以及去除无用边框'

            im_gray = cv2.imread(srcfile, 0)
            x1, x2, y1, y2 = image_preprocess.removeAround(im_gray, 5, f_scale_x=0.2, f_scale_y=0.2)
            im_crop = im_gray[x1:x2 + 1, y1:y2 + 1]
            processedfile = infile + '-1-preprocessed' + '.bmp'
            cv2.imwrite(processedfile, im_crop)
            ## 把原图resize
            im = cv2.imread(srcfile)
            resized_image = cv2.resize(im, (y2 - y1 + 1, x2 - x1 + 1))
            cv2.imwrite(srcfile, resized_image)
            #############################################################################
            ## 第三步：
            ## 使用EdgeBoxes提取候选框
            print '使用EdgeBoxes提取候选框'

            serverIp = '219.224.167.239'
            tcpPort = 9998
            msg = os.path.abspath(processedfile)
            print msg
            print len(msg)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((serverIp, tcpPort))
            time.sleep(1)
            s.send(msg)
            bbsPath = s.recv(512)
            s.close()
            #############################################################################
            bbs = scio.loadmat(bbsPath)
            # print bbs['bbs']
            print bbs['bbs'].shape

            #############################################################################
            ## 第四步：
            ## 使用VGG-16对候选框进行分类
            print '用VGG-16对候选框进行分类'
            tcpPort = 9999
            msg = os.path.abspath(processedfile) + ';;;' + bbsPath
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((serverIp, tcpPort))
            time.sleep(1)
            s.send(msg)
            vgg_res_path = s.recv(512)
            print vgg_res_path
            s.close()

            ## 把结果画在图上
            im = cv2.imread(processedfile)
            vgg_res = np.loadtxt(vgg_res_path, delimiter=',', dtype=float)
            (nboxes, _) = vgg_res.shape
            print 'n_boxes = ', nboxes

            ##
            ## 如果检测结果为空，直接返回
            if len(vgg_res) == 0:
                return render.upload(srcfile, processedfile)
            ##
            im_vgg = im.copy()
            for ib in range(nboxes):
                x1 = np.int16(vgg_res[ib, 0]); y1 = np.int16(vgg_res[ib, 1]);
                x2 = np.int16(vgg_res[ib, 2]); y2 = np.int16(vgg_res[ib, 3]);
                cv2.rectangle(im_vgg, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ##
            vggfile = infile + '-2-vgg' + '.jpg'
            cv2.imwrite(vggfile, im_vgg)
            print 'vggfile = ', vggfile

            #############################################################################
            ## 第五步：
            ## 原始NMS
            print '原始NMS'
            boxes_nms = nms.NMS(vgg_res, threshold=0.3, type='iou')
            # boxes_nms = vgg_res[boxes_nms, :]
            # print boxes_nms
            (nboxes, _) = boxes_nms.shape
            im_nms = im.copy()
            for ib in range(nboxes):
                x1 = np.int16(boxes_nms[ib, 0]); y1 = np.int16(boxes_nms[ib, 1]);
                x2 = np.int16(boxes_nms[ib, 2]); y2 = np.int16(boxes_nms[ib, 3]);
                cv2.rectangle(im_nms, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ##
            nmsfile = infile + '-3-nms' + '.jpg'
            cv2.imwrite(nmsfile, im_nms)
            print 'nmsfile = ', nmsfile

            #############################################################################
            ## 第六步：
            ## 改进的NMS
            print '改进的NMS'
            boxes_nms_modified = nms_modified.nms(vgg_res, boxes_nms, thresh=0.4, weighted=True)
            # boxes_nms = vgg_res[boxes_nms, :]
            # boxes_nms_modified = nms.NMS(boxes_nms_modified, threshold=0.3, type='iou')
            print boxes_nms
            (nboxes, _) = boxes_nms_modified.shape
            im_nms = im.copy()
            for ib in range(nboxes):
                x1 = np.int16(boxes_nms_modified[ib, 0]);
                y1 = np.int16(boxes_nms_modified[ib, 1]);
                x2 = np.int16(boxes_nms_modified[ib, 2]);
                y2 = np.int16(boxes_nms_modified[ib, 3]);
                cv2.rectangle(im_nms, (x1, y1), (x2, y2), (0, 0, 255), 2)
            ##
            nms_modified_img = infile + '-4-nms-modified' + '.jpg'
            cv2.imwrite(nms_modified_img, im_nms)
            print 'nms_modified_file = ', nms_modified_img
            #
            nms_modified_file = filedir + '/' + 'bbox_nms_modified.txt'
            np.savetxt(nms_modified_file, boxes_nms_modified, fmt='%d, %d, %d, %d, %.4f')
            print os.path.abspath(nms_modified_file)
            #############################################################################
            ## 第七步
            ## 结节分类
            print '结节分类'

            ## 首先保存结节

            nodule_dir = filedir + '/' + 'nodules'
            if not os.path.exists(nodule_dir):
                os.mkdir(nodule_dir)
            strRes = ''
            for ib in range(nboxes):
                x1 = np.int16(boxes_nms_modified[ib, 0])
                y1 = np.int16(boxes_nms_modified[ib, 1])
                x2 = np.int16(boxes_nms_modified[ib, 2])
                y2 = np.int16(boxes_nms_modified[ib, 3])
                crop_img = im[y1:y2+1, x1:x2+1]
                nodule_path = os.path.join(nodule_dir, 'nodule_' + str(ib) + '.png')
                cv2.imwrite(nodule_path, crop_img)
                strRes += os.path.abspath(nodule_path) + ' ' + '0' + '\n'
            nodulefile = filedir + '/' + 'nodule_path.txt'
            with open(nodulefile, 'w') as f:
                f.write(strRes)
            print 'nodulefile = ', nodulefile
            ##
            nodulefile_claffication = filedir + '/' + 'nodule_path_classification.txt'
            ## 分类
            tcpPort = 19999
            msg = os.path.abspath(nodulefile) + ';;;' + os.path.abspath(nodulefile_claffication)
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((serverIp, tcpPort))
            time.sleep(1)
            s.send(msg)
            nodulefile_claffication = s.recv(512)
            print nodulefile_claffication
            s.close()

            ## 将结果画在图上
            res_classification = np.loadtxt(nodulefile_claffication, delimiter=',', dtype=float)
            res_classification = res_classification.reshape((-1, 2))
            (nboxes, _) = boxes_nms_modified.shape
            print 'res_classification = ', res_classification
            print 'res_classification.shape = ', res_classification.shape
            ##
            im_classification = im.copy()
            font_size = 1
            text_size = 1
            for ib in range(nboxes):
                x1 = np.int16(boxes_nms_modified[ib, 0]);
                y1 = np.int16(boxes_nms_modified[ib, 1]);
                x2 = np.int16(boxes_nms_modified[ib, 2]);
                y2 = np.int16(boxes_nms_modified[ib, 3]);
                p = res_classification[ib, 1]
                p_str_0 = str("%.2f" % (1 - p))
                p_str_1 = str("%.2f" % p)
                if p >= 0.5:
                    cv2.rectangle(im_classification, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(im_classification, 'p=' + p_str_1, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 0, 255), text_size)
                else:
                    cv2.rectangle(im_classification, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im_classification, 'p=' + p_str_0, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, font_size, (0, 255, 0),text_size)
            ##
            classfile = infile + '-5-classification' + '.jpg'
            cv2.imwrite(classfile, im_classification)
            print 'classfile = ', classfile

            ##
        ## srcfile: 上传之后保存的原图，可能经过resize
        return render.upload(srcfile, processedfile, vggfile, nmsfile, nms_modified_img, classfile)

#!======================================================================================================================
if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()