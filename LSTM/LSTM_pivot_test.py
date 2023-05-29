import tensorflow as tf
from tensorflow import keras
from keras.layers import  LSTM, BatchNormalization, Dense
from LinearRegressionAnalysis import create_convolutional_data
import pandas as pd
from sklearn.model_selection import train_test_split


model = keras.Sequential()
model.add(LSTM(64, input_shape=(2, 21)))
model.add(BatchNormalization())
model.add(Dense(21, activation='softmax'))
print(model.summary())


df = pd.read_csv("../CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
X, y, dates = create_convolutional_data(data=df, historic_data_series=[1,2], month_power=-1, normalize=True)

Wards = df[df.columns[0]].values
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.5, shuffle=False)


X_train = X_train.reshape(X_train.shape[0], 2, 21)
X_test = X_test.reshape(X_test.shape[0], 2, 21)

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train,
          batch_size=64, epochs=5, validation_data=(X_test, y_test))

