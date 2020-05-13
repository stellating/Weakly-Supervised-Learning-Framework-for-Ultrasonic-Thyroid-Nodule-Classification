# Weakly-Supervised-Learning-Framework-for-Ultrasonic-Thyroid-Nodule-Classification
The code of paper "Weakly-Supervised Learning Framework for Ultrasonic Thyroid Nodule Classification"

Subroutines communicate through socket. After all the routines launched, you should input in the web page: http://localhost:8080/upload

Note that you need to specify the IP address and port number in each code (the port number usually does not need to be changed, but it will need to be changed if it is occupied)
The IP address is usually the same as the current IP address

Startup script:
EndMaster_Demo\test.py # main functions, mobilize the whole system, image preprocessing

Receive the image and return to the box extracted from the edge boxes
EdgeBoxes\edges\the start_server.m

Remove most of the non-nodule boxes using vgg16: 
Xception\e0_predictFolder_demo_20181212.py

The original NMS and the improved NMS:
EndMaster_Demo\test.py

For each box, determine the category:
tf_dnn_thyroid_20180124\alexnet_mil_adaptive_20181117\predict_imgs_demo_20181212.py
