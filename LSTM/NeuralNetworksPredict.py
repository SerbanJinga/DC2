import numpy as np
import pandas as pd
import tensorflow as tf
from keras.optimizers import Adam
import seaborn as sns
from Net import getNet

from LinearRegressionAnalysis import create_convolutional_data

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

current_net = 0
turn_into_z_score = False
epochs = 50
batch_size = 5


nets = [0, 1, 2, 3]
lstm_nets = [2, 3]
assert current_net in nets, "Net must be valid net, stored in 'nets' variable"

LSTM = current_net in lstm_nets
convolutional_series = [1] if not LSTM else [1,2,3,4,5,6,7,8,9,10,11,12]
month_power = 2 if not LSTM else -1

"""---DATA AND MODEL CREATION---"""

ward_file = '../CrimeData/Processed/Pivot_December_2012_to_march_2023.csv'
LSOA_file = '../CrimeData/Processed/pivot_LSOA.csv'

df = pd.read_csv(ward_file)
X, y, dates = create_convolutional_data(data=df, historic_data_series=convolutional_series, month_power=month_power, normalize=True)

Wards = df[df.columns[0]].values
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.4, shuffle=False)

normalization_mean = X_train.mean() if turn_into_z_score else 0
normalization_std = X_train.std() if turn_into_z_score else 1

def normalize_into_Zscore(a):
    return (a-normalization_mean)/normalization_std
def denormalize(a):
    return (a*normalization_std)* np.sqrt(len(y_train[0]))

X_train, X_test, y_train, y_test = normalize_into_Zscore(X_train), normalize_into_Zscore(X_test), normalize_into_Zscore(y_train), normalize_into_Zscore(y_test)


if LSTM:
    time_steps = len(convolutional_series)
    features = X_train.shape[1]//time_steps
    X_train = X_train.reshape(X_train.shape[0], time_steps, features)
    X_test = X_test.reshape(X_test.shape[0], time_steps, features)

model = getNet(X_train.shape, y_train.shape, current_net)

optimizer = Adam()#learning_rate=0.00001
model.compile(optimizer=optimizer, loss='mse')
"""---MODEL TRAINING AND EVALUATION---"""


def run_single_model():
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    rmse_unnormalized_loss = np.sqrt(np.array(history.history['loss'], dtype=np.float64))*normalization_std*np.sqrt(len(y_train[0]))
    rmse_unnormalized_val_loss = np.sqrt(np.array(history.history['val_loss'], dtype=np.float64))*normalization_std*np.sqrt(len(y_train[0]))

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

    avg_unnormalized_loss = loss_matrix.mean(axis=0)*normalization_std*np.sqrt(len(y_train[0]))
    avg_unnormalized_val_loss = val_loss_matrix.mean(axis=0)*normalization_std*np.sqrt(len(y_train[0]))

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
    run_model_battery(10)
    #run_single_model()
    current_year = 2023
    # current_month = 1
    # dict_months = {
    #     1: "January",
    #     2: "February",
    #     3: "March",
    #     4: "April",
    #     5: "May",
    #     6: "June",
    #     7: "July",
    #     8: "August",
    #     9: "September",
    #     10: "October",
    #     11: "November",
    #     12: "December"}
    predictions = model.predict(X_test)
    predictions_unscaled = ((predictions * normalization_std) + normalization_mean)
    # ward_prediction = pd.DataFrame({'Wards': Wards, 'Prediction_count': []})
    # print(ward_prediction)
    ward_dict = {}
    for i, ward in enumerate(Wards):
        ward_dict[ward] = predictions_unscaled[:, i][0]*100
        print(f'Ward: {ward}, Predicted Crime Count: {predictions_unscaled[:, i]}')
    ward_predicitions = pd.DataFrame(ward_dict.items())
    ward_predicitions.rename(columns = {0 : "Ward", 1: "Crime Likelihood"}, inplace = True)
    ward_predicitions.set_index("Ward")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Ward',
                     y='Crime Likelihood',
                     data=ward_predicitions)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.xlabel('Ward')
    plt.ylabel('Crime Likelihood')
    plt.title('Crime Likelihood by Ward')
    plt.tight_layout()
    plt.show()
    print(ward_dict)
    print(ward_predicitions)
    print(ward_predicitions['Crime Likelihood'].sum())