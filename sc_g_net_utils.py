import tensorflow as tf
from keras.models import Model
from keras.activations import linear
from keras.layers import Input, Dense,Conv2D, MaxPool2D, \
    AveragePooling2D, Dropout, concatenate, Flatten, Activation
from keras import regularizers
import matplotlib.pyplot as plt

def inception_block(in_filters, out_1x1, out_3x3, out_5x5, maxpool_3x3, l2_value):
    # 1x1 Convolution
    incep_conv_1 = Conv2D(out_1x1, (1,1), padding='same', kernel_regularizer=regularizers.l2(l2_value))(in_filters)
    act_incep_conv_1 = Activation('relu')(incep_conv_1)
    # 3x3 Convolution
    incep_conv_2_1 = Conv2D(out_1x1, (1,1), padding='same', kernel_regularizer=regularizers.l2(l2_value))(in_filters)
    act_incep_conv2_2 = Activation('relu')(incep_conv_2_1)
    incep_conv_2_3 = Conv2D(out_3x3, (3,3), padding='same', kernel_regularizer=regularizers.l2(l2_value))(act_incep_conv2_2)
    act_incep_2_4 =  Activation('relu')(incep_conv_2_3)
    # 5x5 Convolution
    incep_conv_3_1 = Conv2D(out_1x1, (1,1), padding='same', kernel_regularizer=regularizers.l2(l2_value))(in_filters)
    act_incep_conv3_2 = Activation('relu')(incep_conv_3_1)
    incep_conv_3_3 = Conv2D(out_5x5, (5, 5), padding='same', kernel_regularizer=regularizers.l2(l2_value))(act_incep_conv3_2)
    act_incep_3_4 =  Activation('relu')(incep_conv_3_3)
    # 3x3 Maxpool
    incep_maxpool_4_1 = MaxPool2D((3, 3), strides=1, padding='same')(in_filters)
    incep_conv_4_2 = Conv2D(maxpool_3x3, (1,1), padding='same', kernel_regularizer=regularizers.l2(l2_value))(incep_maxpool_4_1)
    act_incep_4_3 =  Activation('relu')(incep_conv_4_2)
    out_filters = concatenate([act_incep_conv_1, act_incep_2_4, act_incep_3_4, act_incep_4_3], axis=-1)
    return out_filters

def aux_layer(x, n_classes):
    x = AveragePooling2D((5, 5), strides=3)(x)
    x = Conv2D(128, (1, 1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(rate=0.7)(x)
    aux_output = Dense(n_classes, activation='softmax')(x)
    return aux_output

def g_net(input_shape, n_cls, l2_value=1.0e-04, desp=False):
    input = Input(input_shape)
    x = Conv2D(64, (7,7), strides=2, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_value), name='l1_conv_depth_1')(input)
    x = MaxPool2D((3, 3), strides=2, padding='same', name='maxpool1_depth_0')(x)
    x = Conv2D(64, (3,3), strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_value), name='l2_conv_depth_1' )(x)
    x = Conv2D(192, (3,3), strides=1, padding='same', activation='relu', kernel_regularizer=regularizers.l2(l2_value), name='l2_conv_depth_2' )(x)
    x = MaxPool2D((3, 3), strides=2, padding='same', name='maxpool2_depth_0')(x)
    x = inception_block(x, 64, 128, 32, 32, l2_value)
    x = inception_block(x, 128, 192, 96, 64, l2_value)
    x = MaxPool2D((3, 3), strides=2, padding='same', name='maxpool3_depth_0')(x)
    x = inception_block(x, 192, 208, 48, 64, l2_value)
    aux_1 = aux_layer(x, n_classes=n_cls)
    x = inception_block(x, 160, 224, 64, 64, l2_value)
    x = inception_block(x, 128, 256, 64, 64, l2_value)
    x = inception_block(x, 112, 288, 64, 64, l2_value)
    aux_2 = aux_layer(x, n_classes=n_cls)
    x = inception_block(x, 256, 320, 128, 128, l2_value)
    x = MaxPool2D((3, 3), strides=2, padding='same', name='maxpool4_depth_0')(x)
    x = inception_block(x, 256, 320, 128, 128, l2_value)
    x = inception_block(x, 384, 384, 128, 128, l2_value)
    x = AveragePooling2D((7, 7), strides=1)(x)
    x = linear(x)
    x = Flatten()(x)
    x = Dropout(rate=0.4)(x)
    x = Dense(n_cls, activation='softmax', kernel_regularizer=regularizers.l2(l2_value), name='output')(x)
    model = Model(input, [x, aux_1, aux_2])
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=3.0e-04,
    decay_steps=2,
    decay_rate=0.05)

    opt = tf.keras.optimizers.Adam(lr_schedule)
    model.compile(loss= 'categorical_crossentropy' , optimizer=opt, metrics=['accuracy'])
    if desp:
        model.summary()
        tf.keras.utils.plot_model(model, show_shapes=True, to_file='sc_googleNet_module.png')
    return model


def show_final_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(15,5))
    ax[0].set_title('loss')
    ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
    ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(history.epoch, history.history["accuracy"], label="Train acc")
    ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation acc")
    ax[0].legend()
    ax[1].legend()
    plt.show()
    plt.savefig('sc_gnet_acccuracy.png')