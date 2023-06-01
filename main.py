import numpy as np

from GeographicConversionHandler import GeographicConversionHandler
from ProcessData import pre_process_raw_data, pivot_table, join_parsed_dataframes
import pandas as pd
import matplotlib.pyplot as plt

from SmallDate import SmallDate
from LinearRegressionAnalysis import create_convolutional_data, trainLinearModel, experiment_results

# Press the green button in the gutter to run the script.



def LinearRegressionBenchmark():
    ward_file = 'CrimeData/Processed/Pivot_December_2012_to_march_2023.csv'
    LSOA_file = 'CrimeData/Processed/pivot_LSOA.csv'
    LSOA_df = pd.read_csv(LSOA_file)
    Ward_df = pd.read_csv(ward_file)

    X_ward, Y_ward, dates_ward = create_convolutional_data(Ward_df, [1], month_power=-1, normalize=True)
    X_lsoa, Y_lsoa, dates_lsoa = create_convolutional_data(LSOA_df, [1], month_power=-1, normalize=True)

    Wards = Ward_df[Ward_df.columns[0]].values
    LSOAs = LSOA_df[LSOA_df.columns[0]].values
    conversion_matrix = GeographicConversionHandler().geo_conv_matrix(Wards, LSOAs)

    print(len(LSOAs))

    def print_rms_test(Delta):
        samples = len(Delta[0])
        trainSamples = int(0.5 * samples)
        rms_test = np.sqrt(np.mean(Delta[:, trainSamples:] ** 2)) * np.sqrt(len(Delta))
        print(rms_test)

    A, Delta, rms_train, rms_test = trainLinearModel(X_lsoa, Y_ward, dates_lsoa, .5)
    experiment_results(A, Delta, rms_train, rms_test)

    """
    print(np.sum(np.abs(Delta)))
    Delta = conversion_matrix@Delta
    print(np.sum(np.abs(Delta)))
    X_predition = conversion_matrix@A@X_lsoa
    Delta = X_predition-Y_ward
    print(np.sum(np.abs(Delta[0])))
    """

def PreviousMonthBenchmark():
    ward_file = 'CrimeData/Processed/Pivot_December_2012_to_march_2023.csv'
    Ward_df = pd.read_csv(ward_file)

    X_ward, Y_ward, dates_ward = create_convolutional_data(Ward_df, [1], month_power=-1, normalize=True)
    def print_rms_test(Delta):
        samples = len(Delta[0])
        trainSamples = int(0.5 * samples)
        rms_test = np.sqrt(np.mean(Delta[:, trainSamples:] ** 2)) * np.sqrt(len(Delta))
        print(rms_test)

    Delta = Y_ward-X_ward
    print_rms_test(Delta)

def PreviousYearBenchmark():
    ward_file = 'CrimeData/Processed/Pivot_December_2012_to_march_2023.csv'
    Ward_df = pd.read_csv(ward_file)

    X_ward, Y_ward, dates_ward = create_convolutional_data(Ward_df, [12], month_power=-1, normalize=True)
    def print_rms_test(Delta):
        samples = len(Delta[0])
        trainSamples = int(0.5 * samples)
        rms_test = np.sqrt(np.mean(Delta[:, trainSamples:] ** 2)) * np.sqrt(len(Delta))
        print(rms_test)

    Delta = Y_ward-X_ward
    print_rms_test(Delta)

if __name__ == '__main__':
    PreviousYearBenchmark()