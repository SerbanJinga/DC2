import numpy as np
import pandas as pd
import tensorflow as tf

from Net import getNet

from LinearRegressionAnalysis import create_convolutional_data

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

current_net = 3
turn_into_z_score = False
epochs = 100
batch_size = 5


nets = [0, 1, 2, 3]
lstm_nets = [2, 3]
assert current_net in nets, "Net must be valid net, stored in 'nets' variable"

LSTM = current_net in lstm_nets
convolutional_series = [1,2] if not LSTM else [1,2,3,4,5,6,7,8,9,10,11,12]
month_power = 2 if not LSTM else -1

"""---DATA AND MODEL CREATION---"""

df = pd.read_csv("../CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
X, y, dates = create_convolutional_data(data=df, historic_data_series=convolutional_series, month_power=month_power, normalize=True)

Wards = df[df.columns[0]].values
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.5, shuffle=False)

normalization_mean = X_train.mean() if turn_into_z_score else 0
normalization_std = X_train.std() if turn_into_z_score else 1

def normalize_into_Zscore(a):
    return (a-normalization_mean)/normalization_std

X_train, X_test, y_train, y_test = normalize_into_Zscore(X_train), normalize_into_Zscore(X_test), normalize_into_Zscore(y_train), normalize_into_Zscore(y_test)


if LSTM:
    time_steps = len(convolutional_series)
    features = X_train.shape[1]//time_steps
    X_train = X_train.reshape(X_train.shape[0], time_steps, features)
    X_test = X_test.reshape(X_test.shape[0], time_steps, features)

model = getNet(X_train.shape, y_train.shape, current_net)

"""---MODEL TRAINING AND EVALUATION---"""


def run_single_model():
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    rmse_unnormalized_loss = np.sqrt(np.array(history.history['loss'], dtype=np.float64))*normalization_std
    rmse_unnormalized_val_loss = np.sqrt(np.array(history.history['val_loss'], dtype=np.float64))*normalization_std

    print(f'rms train: {rmse_unnormalized_loss[-1]}')
    print(f'rms test: {rmse_unnormalized_val_loss[-1]}')

    plt.plot(rmse_unnormalized_loss)
    plt.plot(rmse_unnormalized_val_loss)
    plt.title('Model loss')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def run_model_battery(number_of_times):
    rmse_loss, rmse_val_loss = [0] * number_of_times, [0] * number_of_times

    for n in range(number_of_times):
        print(f'Working with model number {n + 1}')
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0)
        rmse_loss[n] = np.sqrt(np.array(history.history['loss'], dtype=np.float64))
        rmse_val_loss[n] = np.sqrt(np.array(history.history['val_loss'], dtype=np.float64))

    loss_matrix = np.array(rmse_loss)
    val_loss_matrix = np.array(rmse_val_loss)

    avg_unnormalized_loss = loss_matrix.mean(axis=0)*normalization_std
    avg_unnormalized_val_loss = val_loss_matrix.mean(axis=0)*normalization_std

    print(f'rms train avg: {avg_unnormalized_loss[-1]}; std = {np.std(loss_matrix[:, -1])}')
    print(f'rms test avg: {avg_unnormalized_val_loss[-1]}; std = {np.std(val_loss_matrix[:, -1])}')

    plt.plot(avg_unnormalized_loss)
    plt.plot(avg_unnormalized_val_loss)
    plt.title(f'Average model loss, number of models: {number_of_times}')
    plt.ylabel('Average rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # run_model_battery(3)
    run_single_model()
