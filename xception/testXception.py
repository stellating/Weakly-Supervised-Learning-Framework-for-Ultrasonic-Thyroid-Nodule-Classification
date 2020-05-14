#!/usr/bin/env python  
#-*- coding: utf-8 -*-

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, Input
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

img_width = 299; img_height = img_width;
nDense = 512
top_model_weights_path = 'bottleneck_fc_model_' + str(img_width) + '_' + str(nDense) + '.h5'

input_tensor = Input(shape=(img_width, img_height, 3))
base_model = applications.Xception(weights='imagenet', input_tensor = input_tensor, include_top=False)

layers = base_model.layers
for il in range(0, len(layers)):
    layer = layers[il]
    print(il)
    print(layer.input)
    print(layer.output)


exit()

input_tensor = Input(shape=(img_width, img_height, 3))

base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor = input_tensor)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
# top_model.add(Flatten())
top_model.add(Dense(nDense, activation='relu', name='fc1'))
top_model.add(Dropout(0.5))
top_model.add(Dense(nDense, activation='relu', name='fc2'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

layers = model.layers
for il in range(0, len(layers)):
    layer = layers[il]
    print(il)
    print(layer)