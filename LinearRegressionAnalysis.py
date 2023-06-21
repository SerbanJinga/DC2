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
    """
    :param data: pandas dataframe in the form of pivot table with columns corresponding to dates
    and rows corresponding to geographic region

    :param historic_data_series: Array with numbers corresponding to the previous months.
    For example if you want to include the previous month you use [1], if you want the 2 previous months [1,2],
    if you want to predict based on the values a year ago [12], if you want to predict using the whole last year [1,2,3,4,5,6,7,8,9,10,11,12].
    And the degenerate case, where you ask to predict crime based on current crime [0].

    :param month_power: integer corresponding to the maximum power of monthly data.
    The purpose of this is to model non-linear month behaviour, in an attempt to model the seasonality of the information.
    If it is -1 no additional data get's added.
    If it is 0 a constant term gets added to the information to train the model, improving performance.
    If it is 1, you also get the number of the month divided by 12, so for data in april you would also get a value 4/12=0.333...
    For larger number M you would also get additional powers, in the example of april you would get: 1, (4/12), (4/12)^2, ..., (4/12)^M

    :param normalize: this turns dataframe of crime counts per region per time instance into a percentage;
    making the values of every column add up to 1.
    :return: 3 things, first it remembers a numpy matrix with the processed information,
    secondly, it returns a matrix with the desired outputs for each entry
    third, it returns a list of the dates corresponding to each input
    """
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
    """
    Finds the matrix that best fits the data and some metrics.

    :param Phi: is the matrix of inputs of the linear regression
    :param Y: is the matrix of desired outputs that the regression will try to approximate
    :param dates: dates of the inputs, unused parameter
    :param percentTrain: float between 0 and 1, corresponding to how much data to use to fit the model.
    0 means only the first sample, 1 means to train with all the samples,
    0.5 means to train with the first half, and so on...
    The split is NOT randomized, it will always take the first part of the samples.
    :return: first it returns A, the matrix that fits this linear model the best.
    Second, it returns Delta, the matrix of differences between prediction and real values, for all values.
    Thridly, it returns the training error, as in, the error on the data used to fit the model,
    which is the average pythagorean distance between the columns,
    the pythagorean distance is the square root of the sum of squares of differences.
    Lastly, it returns the testing error. Which meansthe error in the data that was NOT used to fit the model.
    """
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
    """
    Renders the results of the experiment to hopefully make them more interpretable.
    :param A: the matrix that was the result of a linear fit. its absolute values will be plotted as a heatmap to see what variables are the most influential.
    :param Delta: the matrix of differences, will be plotted as a time series to visualize how the error changes over time
    :param rms_train: training error, will be printed
    :param rms_test: testing error, will be printed
    :return: None
    """
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
