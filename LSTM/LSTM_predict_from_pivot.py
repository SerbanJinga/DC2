import numpy as np
import pandas as pd
import tensorflow as tf

from Net import getNet

from LinearRegressionAnalysis import create_convolutional_data

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


"""---DATA AND MODEL CREATION---"""

df = pd.read_csv("../CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
X, y, dates = create_convolutional_data(data=df, historic_data_series=[1], month_power=-1, normalize=True)

<<<<<<< HEAD
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
=======
Wards = df[df.columns[0]].values
X_train, X_test, y_train, y_test = train_test_split(X.T, y.T, test_size=0.5, random_state=42)
>>>>>>> 7845647712625cc240b36249054bacd8838947f4

model = getNet(X_train.shape[1], y_train.shape[1], 1)

"""---MODEL TRAINING AND EVALUATION---"""

def run_single_model():
    history = model.fit(X_train, y_train, epochs=200, batch_size=15, validation_data=(X_test, y_test))

<<<<<<< HEAD
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
=======
    rmse_loss = np.sqrt(np.array(history.history['loss'], dtype=np.float64))
    rmse_val_loss = np.sqrt(np.array(history.history['val_loss'], dtype=np.float64))
>>>>>>> 7845647712625cc240b36249054bacd8838947f4

    print(f'rms train: {rmse_loss[-1]}')
    print(f'rms test: {rmse_val_loss[-1]}')


    plt.plot(rmse_loss)
    plt.plot(rmse_val_loss)
    plt.title('Model loss')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


def run_model_battery(number_of_times):
    rmse_loss, rmse_val_loss = [0]*number_of_times, [0]*number_of_times

    for n in range(number_of_times):
        print(f'Working with model number {n+1}')
        history = model.fit(X_train, y_train, epochs=200, batch_size=15, validation_data=(X_test, y_test), verbose=0)
        rmse_loss[n] = np.sqrt(np.array(history.history['loss'], dtype=np.float64))
        rmse_val_loss[n] = np.sqrt(np.array(history.history['val_loss'], dtype=np.float64))

    loss_matrix = np.array(rmse_loss)
    val_loss_matrix = np.array(rmse_val_loss)

    avg_loss = loss_matrix.mean(axis=0)
    avg_val_loss = val_loss_matrix.mean(axis=0)

    print(f'rms train avg: {avg_loss[-1]}; std = {np.std(loss_matrix[:,-1])}')
    print(f'rms test avg: {avg_val_loss[-1]}; std = {np.std(val_loss_matrix[:,-1])}')

    plt.plot(avg_loss)
    plt.plot(avg_val_loss)
    plt.title(f'Average model loss, number of models: {number_of_times}')
    plt.ylabel('Average rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    run_model_battery(5)
