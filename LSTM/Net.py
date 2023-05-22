import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM


def getNet(input_size, output_size, net_id):
    if net_id == 0:
        return getNoLayerSimple(input_size, output_size)
    if net_id == 1:
        return getOneLayerSimple(input_size, output_size)


def getOneLayerSimple(input_size, output_size):
    model = Sequential([
        Dense(10, activation='relu', input_shape=(input_size,)),
        Dense(output_size, activation='sigmoid')])
    model.compile(optimizer='adam', loss='mse')

    return model


def getNoLayerSimple(input_size, output_size):
    model = Sequential([
        Dense(output_size, activation='relu', input_shape=(input_size,))
    ])
    model.compile(optimizer='adam', loss='mse')

    return model
