#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.layers import Conv2D, MaxPooling2D, Activation, Input, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Concatenate, Reshape
from keras.models import Model
from keras.regularizers import l2

def conv_block(x, n_filters, filter_size=3, dropout=0.2):
    x = BatchNormalization()(x)
    x = Activation('relu',)(x)
    x = Conv2D(n_filters, filter_size, padding='same', kernel_initializer='he_uniform')(x)
    if dropout != 0:
        x = Dropout(rate=dropout)(x)
    return x

def down_dense_block(x, growth_rate, blocks, dropout):
    for i in xrange(blocks):
        x1 = conv_block(x, growth_rate, filter_size=3, dropout=dropout)
        x = Concatenate(axis=3)([x, x1])
    return x

def up_dense_block(x, growth_rate, blocks, dropout):
    block_to_upsample = []
    for i in xrange(blocks):
        x1 = conv_block(x, growth_rate, dropout=dropout)
        block_to_upsample.append(x1)
        x = Concatenate(axis=3)([x, x1])
    return Concatenate(axis=3)(block_to_upsample)

def transition_down(x, n_filters, dropout):
    x = conv_block(x, n_filters, filter_size=1, dropout=dropout)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

def transition_up(x, skip_connection, n_filters_keep):

    x = Conv2DTranspose(filters=n_filters_keep, kernel_size=3, strides=(2, 2), padding='same', activation='relu',
                        kernel_initializer='he_uniform')(x)
    x = Concatenate(axis=3)([x, skip_connection])
    return x

def fc_densenet(input_shape=(224,224,3), classes=11, n_filters_first_conv=48, n_pool=4,
             growth_rate=12, n_layers_per_block=5, dropout=0.2):

    if type(n_layers_per_block) == list:
        assert (len(n_layers_per_block) == 2 * n_pool + 1)
    elif type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)
    else:
        raise ValueError

    # first convolution
    inputs = Input(input_shape)
    x = Conv2D(filters=n_filters_first_conv, kernel_size=3, activation='relu', padding='same',
               kernel_initializer='he_uniform')(inputs)
    n_filters = n_filters_first_conv

    # down sample
    skip_connection_list = []

    for i in xrange(n_pool):
        x = down_dense_block(x, growth_rate, n_layers_per_block[i], dropout)
        n_filters += growth_rate*n_layers_per_block[i]
        skip_connection_list.append(x)

        x = transition_down(x, n_filters, dropout)

    skip_connection_list = skip_connection_list[::-1]

    # bottleneck
    x = up_dense_block(x, growth_rate, n_layers_per_block[n_pool], dropout)

    # up sample
    for i in xrange(n_pool):
        x = transition_up(x, skip_connection_list[i], growth_rate * n_layers_per_block[n_pool+i])
        x = up_dense_block(x, growth_rate, n_layers_per_block[n_pool+i+1], dropout)

    # softmax
    x = Conv2D(classes, kernel_size=1, activation='softmax', padding='same', kernel_initializer='he_uniform')(x)
    x = Reshape((input_shape[0]*input_shape[1], classes))(x)
    x = Activation('softmax')(x)

    model = Model(inputs, x)
    model.summary()

    return model


if __name__ == '__main__':
    # fc_densenet(n_pool=5, growth_rate=12, n_layers_per_block=4)
    # fc_densenet(n_pool=5, growth_rate=16, n_layers_per_block=5)
    fc_densenet(n_pool=5, growth_rate=16, n_layers_per_block=[4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4])