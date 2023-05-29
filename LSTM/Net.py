import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, BatchNormalization


def getNet(input_shape, output_shape, net_id):
    if net_id == 0:
        return getNoLayerSimple(input_shape[1], output_shape[1])
    if net_id == 1:
        return getOneLayerSimple(input_shape[1], output_shape[1])
    if net_id == 2:
        return getFirstLSTM(input_shape, output_shape)


def getOneLayerSimple(input_size, output_size):
    model = Sequential([
        Dense(10, activation='relu', input_shape=(input_size,)),
        Dense(output_size, activation='softmax')])
    model.compile(optimizer='adam', loss='mse')

    return model


def getNoLayerSimple(input_size, output_size):
    model = Sequential([
        Dense(output_size, activation='relu', input_shape=(input_size,))
    ])
    model.compile(optimizer='adam', loss='mse')

    return model


def getFirstLSTM(input_shape, output_shape):
    model = Sequential([
        LSTM(64, input_shape=input_shape[1:]),
        #BatchNormalization(),
        Dense(10, activation='relu'),
        Dense(output_shape[1], activation='softmax')
    ])
    model.compile(optimizer='adam', loss='mse')

    return model
