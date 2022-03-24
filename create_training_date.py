import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Lambda
from keras.datasets import cifar10
from keras.preprocessing import image
# from keras.applications.densenet import preprocess_input
import keras.utils.np_utils as kutils

def preporcess_cifar10_data(img_rows, img_columns):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.array([cv2.resize(img, (img_rows,img_columns)) for img in x_train[:,:,:,:]])
    x_test = np.array([cv2.resize(img, (img_rows,img_columns)) for img in x_test[:,:,:,:]])
    y_train = kutils.to_categorical(y_train, 7)
    y_test = kutils.to_categorical(y_test, 7)
    return x_train, y_train, x_test, y_test

def custom_data_gen(trian_datapath, valid_data_path, color_type='rgb'):
    if color_type == 'gray':
        color_type = 'grayscale'

    datagen = image.ImageDataGenerator(rescale=1./255)
    train_dataset = datagen.flow_from_directory(str(trian_datapath), target_size=(224, 224),color_mode=color_type ,batch_size=16, class_mode='categorical')
    valid_dataset = datagen.flow_from_directory(str(valid_data_path), target_size=(224, 224), color_mode=color_type, batch_size=16, class_mode='categorical')

    labels = train_dataset.class_indices
    in_shape = train_dataset.image_shape
    return  train_dataset, valid_dataset, labels, in_shape


