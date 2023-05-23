import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("../CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
ward_encoder = pd.factorize(df['Ward'])
df['Ward'] = ward_encoder[0]
X = df.iloc[:, :-1]  # Exclude the last column (target)
y = df.iloc[:, -1]   # Target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=200, batch_size=512, validation_data=(X_test, y_test))
rmse = model.evaluate(X_test, y_test)
print(f"Model evaluation: {rmse}")
predictions = model.predict(X_test)
actual = []
for i in y_test:
    actual.append(i)
preds = []
for j in predictions:
    preds.append(j[0])

plt.plot(actual, label='Actual value')
plt.plot(preds, label='Predicted value')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Comparison of Two Arrays')
plt.legend()
plt.show()