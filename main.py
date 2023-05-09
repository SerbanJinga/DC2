from ProcessData import pre_process_raw_data, pivot_table, join_parsed_dataframes
import pandas as pd
import matplotlib.pyplot as plt

from SmallDate import SmallDate

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
    pivot_all_data = pd.read_csv("./CrimeData/Processed/Pivot_December_2012_to_march_2023.csv")
    print(pivot_all_data)
