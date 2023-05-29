import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from SmallDate import SmallDate




def getVectorAtTime(data, year, month, normalize=False):
    key = SmallDate(year, month).__str__('-')
    vec = np.array(data[key].values, dtype=np.float64)
    if normalize:
        vec = vec / vec.sum()
    return vec


def create_convolutional_data(data, historic_data_series=[1], month_power=-1, normalize=False):
    Start_date_str, end_date_str = data.columns[2], data.columns[-1]
    Start_date = SmallDate(int(Start_date_str[:4]), int(Start_date_str[-2:])).minus(
        1)  # We reduce Month for the loop, so it includes that end
    End_date = SmallDate(int(end_date_str[:4]), int(end_date_str[-2:]))
    Iter_date = End_date

    Y = []
    Phi = []
    dates = []

    while Start_date.smaller_than_or_equal_to(Iter_date):
        try:

            newY = getVectorAtTime(data, Iter_date.year, Iter_date.month, normalize)
            newConv = []
            for sample in historic_data_series:
                newConv.append(
                    getVectorAtTime(data, Iter_date.minus(sample).year, Iter_date.minus(sample).month, normalize))

            convolution_result = np.array(newConv).flatten()

            month = Iter_date.month
            month_array = np.array([(month/12) ** i for i in range(month_power + 1)], dtype=np.float64).flatten()
            newPhi = np.concatenate((convolution_result, month_array))
            Y.append(newY)
            Phi.append(newPhi)
            dates.append(Iter_date.minus(0))
        except:
            pass
        Iter_date.reduce_month()

    FinalPhi = np.array(Phi).T[:, ::-1]
    FinalY = np.array(Y).T[:, ::-1]


    FinalDates = dates[::-1]
    return FinalPhi,FinalY , FinalDates


def trainLinearModel(Phi, Y, dates, percentTrain):
    samples = len(Y[0])
    trainSamples = int(percentTrain * samples)

    Phi_train = Phi[:, :trainSamples]
    Y_train = Y[:, :trainSamples]
    A = Y_train @ Phi_train.T @ np.linalg.pinv(Phi_train @ Phi_train.T)
    Delta = Y - A @ Phi

    rms_train = np.sqrt(np.mean(Delta[:, :trainSamples] ** 2))*np.sqrt(len(Y))
    rms_test = np.sqrt(np.mean(Delta[:, trainSamples:] ** 2))*np.sqrt(len(Y))

    return A, Delta, rms_train, rms_test


def experiment_results(A, Delta, rms_train, rms_test):
    print(f'rms train: {rms_train}')
    print(f'rms test: {rms_test}')
    figure, axes = plt.subplots(ncols=2)
    ax1, ax2 = axes
    ax1.plot(np.sqrt(np.sum(Delta**2,axis=0)))
    ax2 = sns.heatmap(np.abs(A))
    plt.show()

if __name__ == '__main__':
    ward_file = 'CrimeData/Processed/Pivot_December_2012_to_march_2023.csv'
    LSOA_file = 'CrimeData/Processed/pivot_LSOA.csv'
    main_pivot_data = pd.read_csv(ward_file)

    Phi, Y, dates = create_convolutional_data(main_pivot_data, [1], month_power=-1, normalize=True)
    A, Delta, rms_train, rms_test = trainLinearModel(Phi, Y, dates, .5)

    #print([str(iter) for iter in dates])

    total_crimes = np.array([np.sum(getVectorAtTime(main_pivot_data, Iter_date.year, Iter_date.month)) for Iter_date in dates])

    rms_unnorm = np.sqrt(np.mean((Delta*total_crimes)**2))
    print(f'unnormalized equiv rms: {rms_unnorm}')

    experiment_results(A, Delta, rms_train, rms_test)
