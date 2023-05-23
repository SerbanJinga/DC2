# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import LeakyReLU
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":

    ##  Importing and transposing the transposed_df  ##
    df = pd.read_csv("CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
    df = df.set_index("Ward")
    transposed_df = df.transpose()
    transposed_df = transposed_df.reset_index()
    transposed_df = transposed_df.rename(columns={'index': 'Dates'})
    transposed_df = transposed_df.set_index("Dates")
    print(transposed_df.head())

    dates = transposed_df.index.tolist()
    wards = transposed_df.columns.tolist()

    ##  Defining the LSTM model  ##
    model = Sequential()
    model.add(LSTM(units=64, input_shape=(1, 1)))
    model.add(Dense(units=1, activation='relu'))
    # model.add(LeakyReLU(alpha=0.05))
    model.compile(optimizer='adam', loss='mean_squared_error')

    ward_dict = {}

    for ward in wards:
        # Filter the dataset for the current ward
        ward_data = transposed_df[ward].values

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(ward_data.reshape(-1, 1))

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_data[:-1], scaled_data[1:], test_size=0.5, shuffle=False
        )

        # Reshape the training data for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Train the LSTM model
        model.fit(X_train, y_train, epochs=50, batch_size=35)

        # Predict the likelihood of crimes for the next month
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_likelihood = model.predict(X_test)
        predicted_count = scaler.inverse_transform(predicted_likelihood)

        # Print the predicted count for the next month
        print(f"Ward: {ward}")
        print(f"Predicted Crime Count: {predicted_count[-1][0]}")

        ward_dict[ward] = predicted_count[0][0]

    ##  RMSE  ##
    actual_count = scaler.inverse_transform(y_test)
    rmse = np.sqrt(mean_squared_error(actual_count, predicted_count))
    print(f"RMSE: {rmse}")

    result =pd.DataFrame(ward_dict.items())
    result = result.rename(columns = {0: 'Ward',1: 'predicted_count'})
    result = result.set_index('Ward')
    count = result['predicted_count']
    result["Percentage"] = ((count)/count.sum())*100
    print(result)





