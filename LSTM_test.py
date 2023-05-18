import pandas as pd
from random import randrange
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from GeographicConversionHandler import GeographicConversionHandler
import numpy as np
import tensorflow as tf
import torch

if __name__ == "__main__":
    df = pd.read_csv("CrimeData/Processed/April_2020_to_march_2023.csv")
    LSOA = df['LSOA code'].unique()
    geoHand = GeographicConversionHandler()
    Wards = df['LSOA code'].map(geoHand.LSOA_to_ward()).unique()
    print(len(Wards)) # len of the dataset is now 150000
    print(df.describe())
    df["House price"] = pd.Series([randrange(200000, 500000) for i in range(len(df))])
    dates = pd.to_datetime(df["Month"])
    df["Month"] = dates
    train_size = int(len(df) * 0.8)
    df_train, df_test = df[:train_size], df[train_size:]
    scaler = MinMaxScaler()
    df_train["Month"] = scaler.fit_transform(df_train["Month"].values.reshape(-1, 1))
    df_test["Month"] = scaler.transform(df_test["Month"].values.reshape(-1, 1))
    
    
    X_train = np.reshape(df_train["Month"], (1, 5965, 1))
    y_train = np.reshape(df_train["Month"], (1, 5965, 1))

    X_train = tf.keras.utils.pad_sequences(X_train, dtype="float")
    y_train = tf.keras.utils.pad_sequences(y_train, dtype="float")

    X_train = tf.convert_to_tensor(X_train)
    y_train = tf.convert_to_tensor(y_train)

    X_test = df_test["Month"]

    print(type(X_train), type(y_train))
    # Train the LSTM network
    model = Sequential()
    model.add(LSTM(50, input_shape=(5965, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    hs = model.fit(X_train, y_train, epochs=10, batch_size=1)

    predicted_scaled = model.predict(X_test)
    predicted = scaler.inverse_transform(predicted_scaled)
    print(hs.history)
    print(predicted)