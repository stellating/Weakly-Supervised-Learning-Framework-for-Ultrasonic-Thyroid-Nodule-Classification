# Created on Wed May 31 14:48:46 2017
#
# @author: Frederik Kratzert

"""Containes a helper class for image input pipelines in tensorflow."""

import tensorflow as tf
import numpy as np
import random
from tensorflow import image

#from tensorflow.python.framework import dtypes
#from tensorflow.python.framework.ops import convert_to_tensor

from PIL import Image

# IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)
## 甲状腺超声图像的平均值
IMAGENET_MEAN = tf.constant([80.467195488, 80.467195488, 80.467195488], dtype=tf.float32)

class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, num_classes, class_weight = [1, 1], shuffle=True,
                 buffer_size=1000):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'validation'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            num_classes: Number of classes in the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.

        """
        self.txt_file = txt_file
        self.num_classes = num_classes

        # number of samples in the dataset
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.class_weight = class_weight


        # retrieve the data from the text file
        self.read_txt_file()
        # initial shuffling of the file and label lists (together!)
        self.resetData()

    def set_txt_file(self, train_file):
        self.txt_file = train_file

    def set_class_weight(self, class_weight):
        print('set weight = ' + str(class_weight))
        self.class_weight = class_weight

    def read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            print(self.txt_file + ', lines = ' + str(len(lines)))
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(items[0])
                self.labels.append(int(items[1]))

        self.data_size = len(self.labels)

    def resetData(self):
        print('reset data')
        # initial shuffling of the file and label lists (together!)
        if self.shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = tf.convert_to_tensor(self.img_paths, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.labels, dtype=tf.int32)
        self.class_weight_tensor = tf.convert_to_tensor(self.class_weight, dtype=tf.float32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if self.mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)
            data = data.prefetch(buffer_size=100 * self.batch_size)

        elif self.mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)
            data = data.prefetch(buffer_size=100 * self.batch_size)

        else:
            raise ValueError("Invalid mode '%s'." % (self.mode))

        # shuffle the first `buffer_size` elements of the dataset
        if self.shuffle:
            data = data.shuffle(buffer_size=self.buffer_size)

        # create a new dataset with batches of images
        data = data.batch(self.batch_size)
        self.data = data


    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def _parse_function_train(self, filename, label):
        """Input parser for samples of the training set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)

        weight = tf.reduce_sum(tf.multiply(one_hot, self.class_weight_tensor))
        ### load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        my_img = tf.image.resize_images(img_decoded, [256, 256])

        # tmp = Image.open(filename)
        ################################################################################################################
        """
        Dataaugmentation comes here.        
        """
        # # my_img = PIL_Image.open(filename)
        #
        # # sess = tf.InteractiveSession()
        # # with tf.Session() as sess:
        # # Pass image tensor object to a PIL image
        # image = Image.fromarray(img_resized.eval())
        #
        # # Use PIL or other library of the sort to rotate
        # # rotated = Image.Image.rotate(image, degrees)
        # # Convert rotated image back to tensor
        # # rotated_tensor = tf.convert_to_tensor(np.array(rotated))
        #
        # # my_img = my_img.resize([256, 256]);
        # my_img = Image.fromarray(np.asarray(img_resized).astype(np.uint8))
        # ## 以一半的概率翻转
        # if random.randint(0, 1) == 0:
        #     my_img = my_img.transpose(Image.FLIP_LEFT_RIGHT)
        # ## 旋转
        # rotate_angle_range = [-30, 30]
        # rotate_angle = np.int32(np.random.uniform(rotate_angle_range[0], rotate_angle_range[1]))
        # rotated = Image.Image.rotate(my_img, rotate_angle)


        # ### 转回tensorflow的数据类型
        # my_img = tf.convert_to_tensor(np.array(rotated))
        # ## 随机crop
        my_img = tf.image.random_flip_left_right(my_img, seed=random.randint(0, 10000)) ## 左右随机翻转
        # my_img = image.rotate(my_img, rotate_angle * 3.1415926/180.) ## 随机旋转一定的度数
        my_img = tf.random_crop(my_img, size=[227, 227, 3], seed=random.randint(0, 10000)) ## 随机裁剪
        ################################################################################################################
        ### 转为tf继续处理
        img_centered = tf.subtract(my_img, IMAGENET_MEAN)
        # img_centered = tf.scalar_mul(1./255, img_resized)

        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot, weight

    def _parse_function_inference(self, filename, label):
        """Input parser for samples of the validation/test set."""
        # convert label number into one-hot-encoding
        one_hot = tf.one_hot(label, self.num_classes)
        weight = tf.reduce_sum(tf.multiply(one_hot, self.class_weight_tensor))
        # load and preprocess the image
        img_string = tf.read_file(filename)
        img_decoded = tf.image.decode_png(img_string, channels=3)
        img_resized = tf.image.resize_images(img_decoded, [227, 227])
        img_centered = tf.subtract(img_resized, IMAGENET_MEAN)
        # img_centered = tf.subtract(img_decoded, IMAGENET_MEAN)
        # img_centered = tf.scalar_mul(1./255, img_resized)
        # RGB -> BGR
        img_bgr = img_centered[:, :, ::-1]

        return img_bgr, one_hot, weight
