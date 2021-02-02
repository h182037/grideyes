# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:22:24 2021

@author: sindr
"""

from keras.models import Model
from keras.layers import Input, UpSampling2D, Add, BatchNormalization, Activation, LeakyReLU, add, multiply, Concatenate, Reshape
from keras.regularizers import l2
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, MaxPool2D
from keras.layers.merge import concatenate
import keras.backend as K
import numpy as np

# https://github.com/lixiaolei1982/Keras-Implementation-of-U-Net-R2U-Net-Attention-U-Net-Attention-R2U-Net.-
def up_and_concate(down_layer, layer, data_format='channels_last'):
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    return concate

def rec_res_block(input_layer, out_n_filters, batch_normalization=True, kernel_size=[3, 3], stride=[1, 1],
                  padding='same', data_format='channels_last'):
        input_n_filters = input_layer.get_shape().as_list()[3]
        if out_n_filters != input_n_filters:
            skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding, data_format=data_format)(
                input_layer)
        else:
            skip_layer = input_layer
        layer = skip_layer
        for j in range(2):
            for i in range(2):
                if i == 0:
                    layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                        layer)
                    if batch_normalization:
                        layer1 = BatchNormalization()(layer1)
                    layer1 = Activation('relu')(layer1)
                layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding, data_format=data_format)(
                    add([layer1, layer]))
                if batch_normalization:
                    layer1 = BatchNormalization()(layer1)
                layer1 = Activation('relu')(layer1)
            layer = layer1        
        out_layer = add([layer, skip_layer])
        return out_layer

def attention_block_2d(x, g, inter_channel, data_format='channels_last'):
    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)
    f = Activation('relu')(add([theta_x, phi_g]))
    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)
    rate = Activation('sigmoid')(psi_f)
    att_x = multiply([x, rate])
    return att_x

def attention_up_and_concate(down_layer, layer, data_format='channels_last'):
    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2), data_format=data_format)(down_layer)
    layer = attention_block_2d(x=layer, g=up, inter_channel=in_channel // 4, data_format=data_format)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    return concate

def get_att_unet(shape=(256,256,4), classes=4, data_format='channels_last', dropout=0.2, depth=4, features=64):
    inputs = Input(shape)
    x = Lambda(lambda x: x / 255) (inputs)
    skips = []
    for i in range(depth):
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_last')(x)
        features = features * 2

    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    for i in reversed(range(depth)):
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)

    conv6 = Conv2D(classes, (1, 1), padding='same', data_format=data_format)(x)
    outputs = Conv2D(classes, (1, 1), activation='softmax') (conv6)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_rec_unet(shape=(256,256,4), depth=4, features=64, data_format='channels_last', classes=4):
    inputs = Input(shape)
    x = Lambda(lambda x: x / 255) (inputs)
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

        features = features * 2

    x = rec_res_block(x, features)

    for i in reversed(range(depth)):
        features = features // 2
        x = up_and_concate(x, skips[i])
        x = rec_res_block(x, features)

    conv6 = Conv2D(4, (1, 1), padding='same', data_format=data_format)(x)
    outputs = Conv2D(classes, (1, 1), activation='softmax') (conv6)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def get_UnetPP(num_classes = 4, number_of_filters = 1, kernel = (3,3), shape = (256,256,4), kernel_T = (2,2), drop=None):
    inputs = Input(shape)
    s = Lambda(lambda x: x / 255) (inputs)
    x00 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(s)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    if(drop): x00 = Dropout(0.2)(x00)
    x00 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x00)
    x00 = BatchNormalization()(x00)
    x00 = LeakyReLU(0.01)(x00)
    if(drop): x00 = Dropout(0.2)(x00)
    p0 = MaxPooling2D(pool_size=(2, 2))(x00)
    
    x10 = Conv2D(int(32 * number_of_filters), kernel_size=kernel, padding='same')(p0)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    if(drop): x10 = Dropout(0.2)(x10)
    x10 = Conv2D(int(32 * number_of_filters), kernel_size=kernel, padding='same')(x10)
    x10 = BatchNormalization()(x10)
    x10 = LeakyReLU(0.01)(x10)
    if(drop): x10 = Dropout(0.2)(x10)
    p1 = MaxPooling2D(pool_size=(2, 2))(x10)
    
    x01 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x10)
    x01 = concatenate([x00, x01])
    x01 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    x01 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x01)
    x01 = BatchNormalization()(x01)
    x01 = LeakyReLU(0.01)(x01)
    if(drop): x01 = Dropout(0.2)(x01)
    
    x20 = Conv2D(int(64 * number_of_filters), kernel_size=kernel, padding='same')(p1)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    if(drop): x20 = Dropout(0.2)(x20)
    x20 = Conv2D(int(64 * number_of_filters), kernel_size=kernel, padding='same')(x20)
    x20 = BatchNormalization()(x20)
    x20 = LeakyReLU(0.01)(x20)
    if(drop): x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D(pool_size=(2, 2))(x20)
    
    x11 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x20)
    x11 = concatenate([x10, x11])
    x11 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    x11 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x11)
    x11 = BatchNormalization()(x11)
    x11 = LeakyReLU(0.01)(x11)
    if(drop): x11 = Dropout(0.2)(x11)
    
    x02 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x11)
    x02 = concatenate([x00, x01, x02])
    x02 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    x02 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x02)
    x02 = BatchNormalization()(x02)
    x02 = LeakyReLU(0.01)(x02)
    if(drop): x02 = Dropout(0.2)(x02)
    
    x30 = Conv2D(int(128 * number_of_filters), kernel_size=kernel, padding='same')(p2)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    if(drop): x30 = Dropout(0.2)(x30)
    x30 = Conv2D(int(128 * number_of_filters), kernel_size=kernel, padding='same')(x30)
    x30 = BatchNormalization()(x30)
    x30 = LeakyReLU(0.01)(x30)
    if(drop): x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D(pool_size=(2, 2))(x30)
    
    x21 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x30)
    x21 = concatenate([x20, x21])
    x21 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    x21 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x21)
    x21 = BatchNormalization()(x21)
    x21 = LeakyReLU(0.01)(x21)
    if(drop): x21 = Dropout(0.2)(x21)
    
    x12 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x21)
    x12 = concatenate([x10, x11, x12])
    x12 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    x12 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x12)
    x12 = BatchNormalization()(x12)
    x12 = LeakyReLU(0.01)(x12)
    if(drop): x12 = Dropout(0.2)(x12)
    
    x03 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x12)
    x03 = concatenate([x00, x01, x02, x03])
    x03 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    x03 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x03)
    x03 = BatchNormalization()(x03)
    x03 = LeakyReLU(0.01)(x03)
    if(drop): x03 = Dropout(0.2)(x03)
    
    m = Conv2D(int(256 * number_of_filters), kernel_size=kernel, padding='same')(p3)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    m = Conv2D(int(256 * number_of_filters), kernel_size=kernel, padding='same')(m)
    m = BatchNormalization()(m)
    m = LeakyReLU(0.01)(m)
    if(drop): m = Dropout(0.2)(m)
    
    x31 = Conv2DTranspose(int(128 * number_of_filters), kernel_size=kernel_T, strides=(2,2), padding='same')(m)
    x31 = concatenate([x31, x30])
    x31 = Conv2D(int(128 * number_of_filters), kernel_size=kernel, padding='same')(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    x31 = Conv2D(int(128 * number_of_filters), kernel_size=kernel, padding='same')(x31)
    x31 = BatchNormalization()(x31)
    x31 = LeakyReLU(0.01)(x31)
    if(drop): x31 = Dropout(0.2)(x31)
    
    x22 = Conv2DTranspose(int(64 * number_of_filters), kernel_size=kernel_T, strides=(2,2), padding='same')(x31)
    x22 = concatenate([x22, x20, x21])
    x22 = Conv2D(int(64 * number_of_filters), kernel_size=kernel, padding='same')(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    x22 = Conv2D(int(64 * number_of_filters), kernel_size=kernel, padding='same')(x22)
    x22 = BatchNormalization()(x22)
    x22 = LeakyReLU(0.01)(x22)
    if(drop): x22 = Dropout(0.2)(x22)
    
    x13 = Conv2DTranspose(int(32 * number_of_filters), kernel_size=kernel_T, strides=(2,2), padding='same')(x22)
    x13 = concatenate([x13, x10, x11, x12])
    x13 = Conv2D(int(32 * number_of_filters), kernel_size=kernel, padding='same')(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    x13 = Conv2D(int(32 * number_of_filters), kernel_size=kernel, padding='same')(x13)
    x13 = BatchNormalization()(x13)
    x13 = LeakyReLU(0.01)(x13)
    if(drop): x13 = Dropout(0.2)(x13)
    
    x04 = Conv2DTranspose(int(16 * number_of_filters), kernel_size=kernel_T, strides = (2,2), padding='same')(x13)
    x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
    x04 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    x04 = Conv2D(int(16 * number_of_filters), kernel_size=kernel, padding='same')(x04)
    x04 = BatchNormalization()(x04)
    x04 = LeakyReLU(0.01)(x04)
    if(drop): x04 = Dropout(0.2)(x04)
    
    outputs = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(x04)
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

def get_Unet(act='relu', classes=6, outact='softmax', drop=None, shape=(256,256,4)):
    inputs = Input(shape)
    s = Lambda(lambda x: x / 255) (inputs)
    
    c1 = Conv2D(16, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (s)
    if(drop): c1 = Dropout(drop) (c1)
    c1 = Conv2D(16, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)
    
    c2 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p1)
    if(drop): c2 = Dropout(drop) (c2)
    c2 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)
    
    c3 = Conv2D(64, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p2)
    if(drop): c3 = Dropout(drop) (c3)
    c3 = Conv2D(64, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)
    
    c4 = Conv2D(128, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p3)
    if(drop): c4 = Dropout(drop) (c4)
    c4 = Conv2D(128, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    
    c5 = Conv2D(256, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (p4)
    if(drop): c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c5)
    
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u6)
    if(drop): c6 = Dropout(drop) (c6)
    c6 = Conv2D(128, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c6)
    
    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u7)
    if(drop): c7 = Dropout(drop) (c7)
    c7 = Conv2D(64, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c7)
    
    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u8)
    if(drop): c8 = Dropout(drop) (c8)
    c8 = Conv2D(32, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c8)
    
    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (u9)
    if(drop): c9 = Dropout(drop) (c9)
    c9 = Conv2D(16, (3, 3), activation=act, kernel_initializer='he_normal', padding='same') (c9)
    
    outputs = Conv2D(classes, (1, 1), activation=outact) (c9)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model

# https://github.com/guilherme-pombo/keras_resnext_fpn/blob/master/resnext_fpn.py
def get_resFPN(input_shape, nb_labels, depth=(3, 4, 6, 3), cardinality=32, width=4, weight_decay=5e-4, batch_norm=True, batch_momentum=0.9, pyramid=256):
    """
    TODO: add dilated convolutions as well
    Resnext-50 is defined by (3, 4, 6, 3) [default]
    Resnext-101 is defined by (3, 4, 23, 3)
    Resnext-152 is defined by (3, 8, 23, 3)
    :param input_shape:
    :param nb_labels:
    :param depth:
    :param cardinality:
    :param width:
    :param weight_decay:
    :param batch_norm:
    :param batch_momentum:
    :return:
    """
    nb_rows, nb_cols, _ = input_shape
    input_tensor = Input(shape=input_shape)

    bn_axis = 3
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1', kernel_regularizer=l2(weight_decay))(input_tensor)
    if batch_norm:
        x = BatchNormalization(axis=bn_axis, name='bn_conv1', momentum=batch_momentum)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    stage_1 = x

    # filters are cardinality * width * 2 for each depth level
    for i in range(depth[0]):
        x = bottleneck_block(x, 128, cardinality, strides=1, weight_decay=weight_decay)
    stage_2 = x

    # this can be done with a for loop but is more explicit this way
    x = bottleneck_block(x, 256, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[1]):
        x = bottleneck_block(x, 256, cardinality, strides=1, weight_decay=weight_decay)
    stage_3 = x

    x = bottleneck_block(x, 512, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[2]):
        x = bottleneck_block(x, 512, cardinality, strides=1, weight_decay=weight_decay)
    stage_4 = x

    x = bottleneck_block(x, 1024, cardinality, strides=2, weight_decay=weight_decay)
    for idx in range(1, depth[3]):
        x = bottleneck_block(x, 1024, cardinality, strides=1, weight_decay=weight_decay)
    stage_5 = x

    P5 = Conv2D(pyramid, (1, 1), name='fpn_c5p5')(stage_5)
    P4 = Add(name="fpn_p4add")([UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
                                Conv2D(pyramid, (1, 1), name='fpn_c4p4', padding='same')(stage_4)])
    P3 = Add(name="fpn_p3add")([UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
                                Conv2D(pyramid, (1, 1), name='fpn_c3p3')(stage_3)])
    P2 = Add(name="fpn_p2add")([UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
                                Conv2D(pyramid, (1, 1), name='fpn_c2p2', padding='same')(stage_2)])
    # Attach 3x3 conv to all P layers to get the final feature maps. --> Reduce aliasing effect of upsampling
    P2 = Conv2D(pyramid, (3, 3), padding="SAME", name="fpn_p2")(P2)
    P3 = Conv2D(pyramid, (3, 3), padding="SAME", name="fpn_p3")(P3)
    P4 = Conv2D(pyramid, (3, 3), padding="SAME", name="fpn_p4")(P4)
    P5 = Conv2D(pyramid, (3, 3), padding="SAME", name="fpn_p5")(P5)

    head1 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head1_conv")(P2)
    head1 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head1_conv_2")(head1)

    head2 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head2_conv")(P3)
    head2 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head2_conv_2")(head2)

    head3 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head3_conv")(P4)
    head3 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head3_conv_2")(head3)

    head4 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head4_conv")(P5)
    head4 = Conv2D(pyramid // 2, (3, 3), padding="SAME", name="head4_conv_2")(head4)

    f_p2 = UpSampling2D(size=(8, 8), name="pre_cat_2")(head4)
    f_p3 = UpSampling2D(size=(4, 4), name="pre_cat_3")(head3)
    f_p4 = UpSampling2D(size=(2, 2), name="pre_cat_4")(head2)
    f_p5 = head1

    x = Concatenate(axis=-1)([f_p2, f_p3, f_p4, f_p5])
    x = Conv2D(nb_labels, (3, 3), padding="SAME", name="final_conv", kernel_initializer='he_normal',
               activation='linear')(x)
    x = UpSampling2D(size=(4, 4), name="final_upsample")(x)
    x = Activation('sigmoid')(x)

    model = Model(input_tensor, x)

    return model


def grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        return x

    for c in range(cardinality):
        x = Lambda(lambda z: z[:, :, :, c * grouped_channels:(c + 1) * grouped_channels])(input)
        x = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False, strides=(strides, strides),
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(x)
        group_list.append(x)

    group_merge = concatenate(group_list, axis=3)
    x = BatchNormalization(axis=3)(group_merge)
    x = Activation('relu')(x)
    return x


def bottleneck_block(input, filters=64, cardinality=8, strides=1, weight_decay=5e-4):
    init = input
    grouped_channels = int(filters / cardinality)

    if init._keras_shape[-1] != 2 * filters:
        init = Conv2D(filters * 2, (1, 1), padding='same', strides=(strides, strides),
                      use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(init)
        init = BatchNormalization(axis=3)(init)

    x = Conv2D(filters, (1, 1), padding='same', use_bias=False,
               kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)

    x = grouped_convolution_block(x, grouped_channels, cardinality, strides, weight_decay)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization(axis=3)(x)

    x = add([init, x])
    x = Activation('relu')(x)
    return x
# https://github.com/junyuchen245/SPECT-CT-Seg-ResUNet-Keras/blob/master/nets/resUnet.py
def get_ResUnet(pretrained_weights = None, input_size = (256,256,4)):

    """ first encoder for spect image """
    input_seg = Input(input_size)
    input_segBN = BatchNormalization()(input_seg)

    conv1_spect = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_segBN)
    conv1_spect = BatchNormalization()(conv1_spect)
    conv1_spect = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_spect)
    conv1_spect = BatchNormalization(name='conv_spect_32')(conv1_spect)
    conv1_spect = Add()([conv1_spect, input_segBN])
    pool1_spect = MaxPool2D(pool_size=(2, 2))(conv1_spect)


    conv2_spect_in = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_spect)
    conv2_spect_in = BatchNormalization()(conv2_spect_in)
    conv2_spect = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_spect_in)
    conv2_spect = BatchNormalization(name='conv_spect_64')(conv2_spect)
    conv2_spect = Add()([conv2_spect, conv2_spect_in])
    pool2_spect = MaxPool2D(pool_size=(2, 2))(conv2_spect)

    conv3_spect_in = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_spect)
    conv3_spect_in = BatchNormalization()(conv3_spect_in)
    conv3_spect = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_spect_in)
    conv3_spect = BatchNormalization(name='conv_spect_128')(conv3_spect)
    conv3_spect = Add()([conv3_spect, conv3_spect_in])
    pool3_spect = MaxPool2D(pool_size=(2, 2))(conv3_spect)

    conv4_spect_in = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_spect)
    conv4_spect_in = BatchNormalization()(conv4_spect_in)
    conv4_spect = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_spect_in)
    conv4_spect = BatchNormalization(name='conv_spect_256')(conv4_spect)
    conv4_spect = Add()([conv4_spect, conv4_spect_in])
    drop4_spect = Dropout(0.5)(conv4_spect)
    pool4_spect = MaxPool2D(pool_size=(2, 2))(drop4_spect)

    """ second encoder for ct image """
    input_ct = Input(input_size)
    input_ctBN = BatchNormalization()(input_ct)

    conv1_ct = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input_ctBN)
    conv1_ct = BatchNormalization()(conv1_ct)
    conv1_ct = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1_ct)
    conv1_ct = BatchNormalization(name='conv_ct_32')(conv1_ct)
    conv1_ct = Add()([conv1_ct, input_ctBN])
    pool1_ct = MaxPool2D(pool_size=(2, 2))(conv1_ct) #192x192

    conv2_ct_in = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1_ct)
    conv2_ct_in = BatchNormalization()(conv2_ct_in)
    conv2_ct = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2_ct_in)
    conv2_ct = BatchNormalization(name='conv_ct_64')(conv2_ct)
    conv2_ct = Add()([conv2_ct, conv2_ct_in])
    pool2_ct = MaxPool2D(pool_size=(2, 2))(conv2_ct) #96x96

    conv3_ct_in = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2_ct)
    conv3_ct_in = BatchNormalization()(conv3_ct_in)
    conv3_ct = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3_ct_in)
    conv3_ct = BatchNormalization(name='conv_ct_128')(conv3_ct)
    conv3_ct = Add()([conv3_ct, conv3_ct_in])
    pool3_ct = MaxPool2D(pool_size=(2, 2))(conv3_ct) #48x48

    conv4_ct_in = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3_ct)
    conv4_ct_in = BatchNormalization()(conv4_ct_in)
    conv4_ct = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4_ct_in)
    conv4_ct = BatchNormalization(name='conv_ct_256')(conv4_ct)
    conv4_ct = Add()([conv4_ct, conv4_ct_in])
    drop4_ct = Dropout(0.5)(conv4_ct)
    pool4_ct = MaxPool2D(pool_size=(2, 2))(drop4_ct) #24x24 

    conv5_ct_in = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_ct)
    conv5_ct_in = BatchNormalization()(conv5_ct_in)
    conv5_ct = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_ct_in)
    conv5_ct = BatchNormalization(name='conv_ct_512')(conv5_ct)
    conv5_ct = Add()([conv5_ct, conv5_ct_in])
    conv5_ct = Dropout(0.5)(conv5_ct)
    #pool5_ct = MaxPool2D(pool_size=(2, 2))(conv5_ct) #12x12

    conv5_spect_in = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4_spect)
    conv5_spect_in = BatchNormalization()(conv5_spect_in)
    conv5_spect = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_spect_in)
    conv5_spect = BatchNormalization(name='conv_spect_512')(conv5_spect)
    conv5_spect = Add()([conv5_spect, conv5_spect_in])
    conv5_spect = Dropout(0.5)(conv5_spect)
    #pool5_spect = MaxPool2D(pool_size=(2, 2))(conv5_spect)

    merge5_cm = concatenate([conv5_spect, conv5_ct], axis=3) #12x12

    up7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(merge5_cm)) #24x24
    up7_cm = BatchNormalization()(up7_cm)
    merge7_cm = concatenate([drop4_ct, drop4_spect, up7_cm], axis=3)  # cm: cross modality
    merge7_cm = BatchNormalization()(merge7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7_cm)
    conv7_cm_in = BatchNormalization()(conv7_cm)
    conv7_cm = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7_cm_in)
    conv7_cm = BatchNormalization(name='decoder_conv_256')(conv7_cm)
    conv7_cm = Add()([conv7_cm, conv7_cm_in])

    up8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7_cm))
    up8_cm = BatchNormalization()(up8_cm)
    merge8_cm = concatenate([conv3_ct, conv3_spect, up8_cm], axis=3)  # cm: cross modality
    merge8_cm = BatchNormalization()(merge8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8_cm)
    conv8_cm_in = BatchNormalization()(conv8_cm)
    conv8_cm = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8_cm_in)
    conv8_cm = BatchNormalization(name='decoder_conv_128')(conv8_cm)
    conv8_cm = Add()([conv8_cm, conv8_cm_in])

    up9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8_cm))
    up9_cm = BatchNormalization()(up9_cm)
    merge9_cm = concatenate([conv2_ct, conv2_spect, up9_cm], axis=3)  # cm: cross modality
    merge9_cm = BatchNormalization()(merge9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9_cm)
    conv9_cm_in = BatchNormalization()(conv9_cm)
    conv9_cm = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9_cm_in)
    conv9_cm = BatchNormalization(name='decoder_conv_64')(conv9_cm)
    conv9_cm = Add()([conv9_cm, conv9_cm_in])

    up10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv9_cm))
    up10_cm = BatchNormalization()(up10_cm)
    merge10_cm = concatenate([conv1_ct, conv1_spect, up10_cm], axis=3)  # cm: cross modality
    merge10_cm = BatchNormalization()(merge10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10_cm)
    conv10_cm_in = BatchNormalization()(conv10_cm)
    conv10_cm = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10_cm_in)
    conv10_cm = BatchNormalization(name='decoder_conv_32')(conv10_cm)
    conv10_cm = Add()([conv10_cm, conv10_cm_in])

    conv11_cm = Conv2D(filters=6, kernel_size=3, activation='relu', padding='same')(conv10_cm)
    conv11_cm = BatchNormalization()(conv11_cm)
    out = Conv2D(filters=3, kernel_size=1, activation='softmax', padding='same', name='segmentation')(conv11_cm)
    image_size = tuple((256, 256))

    x = Reshape((np.prod(image_size), 3))(out)

    model = Model(inputs=[input_ct, input_seg], outputs=x)

