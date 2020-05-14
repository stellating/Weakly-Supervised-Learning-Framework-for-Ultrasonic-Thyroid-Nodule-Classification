#!/usr/bin/env python
#-*- coding: utf-8 -*-

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model, Input
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

# dimensions of our images.
img_width, img_height = 75, 75
# path to the model weights files.

nTrain_nodule = 63540;
# nTrain_nodule = 1842;
nTrain_nonnodule = nTrain_nodule * 5;
nValidation_nodule = 2302;
nValidation_nonnodule = nValidation_nodule;

train_data_dir = 'I:/thyroid-data-all-gray-20171224/keras/data-' + str(nTrain_nodule) + '-' + str(nTrain_nonnodule) + '-' + str(img_width) + '/train'
validation_data_dir = 'I:/thyroid-data-all-gray-20171224/keras/data-' + str(nTrain_nodule) + '-' + str(nTrain_nonnodule) + '-' + str(img_width) + '/validation'

nb_train_samples = nTrain_nodule + nTrain_nonnodule;
nb_validation_samples = nValidation_nodule + nValidation_nonnodule;

nDense = 512
top_model_weights_path = 'bottleneck_fc_model_' + str(img_width) + '_' + str(nDense) + '.h5'

epochs = 500
batch_size = 64
batch_size_validation = 64;
countFreezes = [0];
class_weight = {0:5, 1:1}

##
init_lr = 0.01;
factor_lr = 0.5
patience_lr = 5
patience_es = 20

for iii in range(0, len(countFreezes)):
    countFreeze = countFreezes[iii];
    print('countFreeze = ' + str(countFreeze))

    # build the VGG16 network
    input_tensor = Input(shape=(img_width, img_height, 3))
    base_model = applications.Xception(weights='imagenet', include_top=False, input_tensor = input_tensor)
    print('Model loaded.')

    top_model = Sequential()
    top_model.add(GlobalAveragePooling2D(input_shape=base_model.output_shape[1:]))
    base_model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    # top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(nDense, activation='relu', name='fc1', input_shape=base_model.output_shape[1:]))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(nDense, activation='relu', name='fc2'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained
    # classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)

    layers = model.layers
    print('model.layers = ' + str(len(layers)))
    for il in range(0, len(layers)):
        layer = layers[il]
        print(il)
        print(layer)
    # exit(0)
    for layer in model.layers[:countFreeze]:
        layer.trainable = False
    ##

    # compile the model with a SGD/momentum optimizer
    # and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=init_lr, momentum=0.9),
                  metrics=['accuracy'],
                  )

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=[0.9, 1.1],
        rotation_range=45,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0,
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size_validation,
        class_mode='binary')

    # call back
    checkpointer = ModelCheckpoint(filepath='my_model_' + str(countFreeze) + '_' + str(img_width) + '_best.h5', save_best_only=True)
    tensorBoard = TensorBoard(log_dir='d:/keras_logs');
    reduceLRCallBack = ReduceLROnPlateau(monitor='val_loss', factor=factor_lr, patience=patience_lr, verbose=1, mode='auto', epsilon=0.0001,
                                         cooldown=0,
                                         min_lr=0)
    earlyStopCallBack = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')

    # fine-tune the model
    model.fit_generator(
        train_generator,
        # samples_per_epoch=nb_train_samples,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs=epochs,
        validation_data = validation_generator,
        validation_steps= nb_validation_samples // batch_size_validation,
        callbacks = [checkpointer, tensorBoard, reduceLRCallBack, earlyStopCallBack],
        verbose = 2,
        class_weight = class_weight,
    )

    model.save('my_model_' + str(countFreeze) + '_' + str(img_width) + '_final.h5')