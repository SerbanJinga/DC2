import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from random import randrange
from GeographicConversionHandler import GeographicConversionHandler
from sklearn.metrics import mean_squared_error
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv("CrimeData/Processed/january_2022_to_march_2023.csv")
    LSOA = df['LSOA code'].unique()
    geoHand = GeographicConversionHandler()
    Wards = df['LSOA code'].map(geoHand.LSOA_to_ward()).unique()
    df["AverageHousePrice"] = pd.Series([randrange(200000, 500000) for i in range(len(df))])
    df['Month'] = pd.to_datetime(df['Month'])
    df = df.set_index('Month')
    train_size = int(len(df) * 0.8)  # 80% training df, 20% testing df
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    # Step 4: Normalize df
    scaler = MinMaxScaler()
    df_train_scaled = scaler.fit_transform(df_train[['AverageHousePrice']])
    df_test_scaled = scaler.transform(df_test[['AverageHousePrice']])

    # Step 5: Create Sequences
    def create_sequences(df, seq_length):
        X, y = [], []
        for i in range(len(df) - seq_length):
            X.append(df[i:i+seq_length])
            y.append(df[i+seq_length])
        return np.array(X), np.array(y)

    seq_length = 3  # Adjust the sequence length as needed
    X_train, y_train = create_sequences(df_train_scaled, seq_length)
    X_test, y_test = create_sequences(df_test_scaled, seq_length)

    model = Sequential()
    model.add(LSTM(64, input_shape=(seq_length, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    print(f"Train RMSE: {train_rmse}")
    print(f"Test RMSE: {test_rmse}")

    future_df = df[['AverageHousePrice']].tail(seq_length)
    future_df_scaled = scaler.transform(future_df)
    X_future = np.array([future_df_scaled])
    future_predictions = model.predict(X_future)
    predicted_house_price = scaler.inverse_transform(future_predictions)
    print(f"Predicted average house price for the next month: {predicted_house_price[0][0]}")
