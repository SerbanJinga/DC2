from ProcessData import pre_process_raw_data, pivot_table
import pandas as pd
# Press the green button in the gutter to run the script.



# First we access data

burglary_data_April_2020_to_march_2023 = None
try:
    burglary_data_April_2020_to_march_2023 = pd.read_csv('CrimeData/Processed/April_2020_to_march_2023.csv')
except:
    # if the data has not been stored, we process it, store it, and access it
    pre_process_raw_data()
    burglary_data_April_2020_to_march_2023 = pd.read_csv('CrimeData/Processed/April_2020_to_march_2023.csv')


if __name__ == '__main__':

    # test
    print(len(burglary_data_April_2020_to_march_2023['Ward'].unique())==21)
    # Pivot
    print("Will turn data to pivot")
    print('='*50)
    print(pivot_table(burglary_data_April_2020_to_march_2023))
    print('=' * 50)

