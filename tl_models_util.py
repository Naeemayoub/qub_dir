from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras.models import Model
import tensorflow as tf

def make_inception_v3_model(n_classes, in_shape):
    input = Input(in_shape)
    base_model = InceptionV3(input_tensor=input, include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    # for l in model.layers:
    #     print(l.name, l.trainable)
    model.compile(
        optimizer= tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3.0e-04, decay_steps=439, decay_rate=0.5, staircase=False), beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    return model


def make_xception_model(n_classes, in_shape):
    input = Input(in_shape)
    base_model = Xception(include_top=False, input_tensor=input)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.4)(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer= tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3.0e-04, decay_steps=439, decay_rate=0.5, staircase=False), beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
    return model
