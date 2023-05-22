import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from LinearRegressionAnalysis import create_convolutional_data

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("../CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
X, y, dates = create_convolutional_data(data=df, historic_data_series=[1], month_power=-1, normalize=True)

Wards = df[df.columns[0]].values

X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.5, random_state=42)

# I already normalize the data when loading
# mean = X_train.mean(axis=0)
# std = X_train.std(axis=0)
# X_train = (X_train - mean) / std
# X_test = (X_test - mean) / std


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    #tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(y_train.shape[1], activation='relu')
])

model.compile(optimizer='adam', loss='mse')

history = model.fit(X_train, y_train, epochs=100, batch_size=15, validation_data=(X_test, y_test))
rmse = np.sqrt(model.evaluate(X_test, y_test))
print(f"Model evaluation rmse: {rmse}")


predictions = model.predict(X_test)

Delta = predictions-y_test

rmse_loss = np.sqrt(np.array(history.history['loss'], dtype=np.float64))
rmse_val_loss = np.sqrt(np.array(history.history['val_loss'], dtype=np.float64))

plt.plot(rmse_loss)
plt.plot(rmse_val_loss)
plt.title('Model loss')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


"""
plt.bar(Wards, np.abs(Delta).mean(axis=0), label='Mean Absolute Error')
plt.xlabel('Ward')
plt.ylabel('Value')
plt.title('Mean error per ward')
plt.legend()
plt.show()
"""
