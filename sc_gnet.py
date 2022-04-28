from __future__ import print_function
from matplotlib.pyplot import axes
import tensorflow as tf
import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, ZeroPadding2D, Flatten, concatenate, Reshape, Activation
from keras.models import Model
from keras.regularizers import l2

l2_value = 1.0e-04


def inception_block(in_filters, in_1x1, in_3x3_reduced, in_3x3, in_5x5_reduced, in_5x5, in_1x1_maxpool, l2_value):
    # 1x1 Convolution
    incep_conv_1_1 = Conv2D(in_1x1, (1,1), strides=(1,1), padding='same', activation='relu', kernel_regularizer=l2(l2_value))(in_filters)
    # 3x3 Convolution
    incep_conv_2_1 = Conv2D(in_3x3_reduced, (1,1), padding='same', activation='relu', kernel_regularizer=l2(l2_value))(in_filters)
    incep_conv_2_2 = Conv2D(in_3x3, (3,3), padding='same', activation='relu', kernel_regularizer=l2(l2_value))(incep_conv_2_1)

    # 5x5 Convolution
    incep_conv_3_1 = Conv2D(in_5x5_reduced, (1,1), padding='same', activation='relu', kernel_regularizer=l2(l2_value))(in_filters)
    incep_conv_3_2 = Conv2D(in_5x5, (5,5), padding='same', activation='relu', kernel_regularizer=l2(l2_value))(incep_conv_3_1)
    # 3x3 Maxpool
    incep_maxpool_4_1 = MaxPooling2D((3, 3), strides=(1,1), padding='same')(in_filters)
    incep_conv_4_2 = Conv2D(in_1x1_maxpool, (1,1), padding='same', kernel_regularizer=l2(l2_value))(incep_maxpool_4_1)
    in_filters = concatenate([incep_conv_1_1, incep_conv_2_2, incep_conv_3_2, incep_conv_4_2], axis=-1)
    return in_filters

def aux_layer (x, n_classes):
    x1 = AveragePooling2D((5, 5), strides=3)(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(n_classes, activation='softmax')(x1)
    return x1
def create_googlenet(n_classes, shape):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    input = Input(shape=shape)
    x = Conv2D(64, (7,7), strides=(2,2), padding='same', activation='relu', name='conv1/7x7', kernel_regularizer=l2(l2_value))(input)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool1/3x3')(x)
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu', name='conv2/3x3_reduced', kernel_regularizer=l2(l2_value))(x)
    x = Conv2D(192, (3,3), strides=(1,1), padding='same', activation='relu', name='conv2/3x3', kernel_regularizer=l2(l2_value))(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool2/3x3')(x)
    inception3a = inception_block(x, 64, 96, 128, 16, 32, 32, l2_value)
    inception3b = inception_block(inception3a, 128, 128, 192, 32, 96, 64, l2_value)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool3/3x3')(inception3b)
    inception4a = inception_block(x, 192, 96, 208, 16, 48, 64, l2_value)
    aux_layer_1 = aux_layer(inception4a, n_classes=n_classes)
    inception4b = inception_block(inception4a, 160, 112, 224, 24, 64, 64, l2_value)
    inception4c = inception_block(inception4b, 128, 128, 256, 24, 64, 64, l2_value)
    inception4d = inception_block(inception4c, 112, 144, 288, 32, 64, 64, l2_value)
    aux_layer_2 = aux_layer(inception4d, n_classes=n_classes)
    x = inception_block(inception4d, 256, 160, 320, 32, 128, 128, l2_value)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='pool4/3x3')(x)
    inception5a = inception_block(x, 256, 160, 320, 32, 128, 128, l2_value)
    x = inception_block(inception5a, 384, 192, 384, 48, 128, 128, l2_value)
    
    x = AveragePooling2D(pool_size=(7,7), strides=(1,1), name='ave_pool1/7x7')(x)
    x = Flatten()(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(n_classes, activation='softmax', kernel_regularizer=l2(l2_value))(x)
    googlenet = Model(inputs=input, outputs=x)
    # googlenet = Model(inputs=input, inputs=[loss1_classifier_act,loss2_classifier_act,loss3_classifier_act])
    googlenet.compile(
        optimizer= tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3.0e-04, decay_steps=439, decay_rate=0.5, staircase=False), beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    return googlenet