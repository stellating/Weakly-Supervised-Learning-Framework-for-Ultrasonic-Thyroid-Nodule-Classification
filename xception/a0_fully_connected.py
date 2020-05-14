#!/usr/bin/env python
#-*- coding: utf-8 -*-

import numpy as np
import sys
sys.path.append('../data')
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Input, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications, optimizers
from keras.callbacks import LambdaCallback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# dimensions of our images.

img_width, img_height = 75, 75

nTrain_nodule = 1842;
nTrain_nonnodule = nTrain_nodule * 1;
nValidation_nodule = 1842;
nValidation_nonnodule = nValidation_nodule;

train_data_dir = 'I:/thyroid-data-all-gray-20171224/keras/data-' + str(nTrain_nodule) + '-' + str(nTrain_nonnodule) + '-' + str(img_width) + '/train'
validation_data_dir = 'I:/thyroid-data-all-gray-20171224/keras/data-' + str(nTrain_nodule) + '-' + str(nTrain_nonnodule) + '-' + str(img_width) + '/validation'

nb_train_samples = nTrain_nodule + nTrain_nonnodule;
nb_validation_samples = nValidation_nodule + nValidation_nonnodule;

nDense = 512

top_model_weights_path = 'bottleneck_fc_model_' + str(img_width) + '_' + str(nDense) + '.h5'
epochs = 200
batch_size = 64
class_weight = {0:1, 1:1}

colorChannels = 3;
colorMode = 'rgb'
if colorChannels == 1:
    colorMode = 'grayscale'

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    input_tensor = Input(shape=(img_width, img_height, colorChannels))
    base_model = applications.Xception(include_top=False, weights='imagenet', input_tensor = input_tensor,
                                    input_shape=(img_width, img_height, colorChannels))

    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # model = base_model
    #
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        color_mode=colorMode,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train_' + str(img_width) + '_' + str(nDense) + '.npy', 'wb'),
            bottleneck_features_train)
    print('bottleneck_features_train.shape')
    print(bottleneck_features_train.shape)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        color_mode=colorMode,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation_' + str(img_width) + '_' + str(nDense) + '.npy', 'wb'),
            bottleneck_features_validation)

def train_top_model():
    train_data = np.load(open('bottleneck_features_train_' + str(img_width) + '_' + str(nDense) + '.npy', 'rb'))
    train_labels = np.array(
        [0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation_' + str(img_width) + '_' + str(nDense) + '.npy', 'rb'))
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2))

    model = Sequential()
    # model.add(Flatten(input_shape=train_data.shape[1:]))

    model.add(Dense(nDense, activation='relu', name='fc1', input_shape=train_data.shape[1:]))
    model.add(Dropout(0.5))
    model.add(Dense(nDense, activation='relu', name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        # optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'],
                  optimizer=optimizers.SGD(lr=1e-2),
                  )

    reduceLRCallBack = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience=30, verbose=1, mode='auto', epsilon=0.0001,
                                         cooldown=0, min_lr=0)
    earlyStopCallBack = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')

    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=top_model_weights_path, save_best_only=True, save_weights_only=True)

    model.fit(train_data, train_labels,
              epochs=epochs,
              verbose=2,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[reduceLRCallBack, earlyStopCallBack, checkpointer],
              class_weight=class_weight,
              )

    # model.save_weights(top_model_weights_path)

save_bottlebeck_features()
train_top_model()
