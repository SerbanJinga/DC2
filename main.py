from ProcessData import pre_process_raw_data, pivot_table, join_parsed_dataframes
import pandas as pd
import matplotlib.pyplot as plt

from SmallDate import SmallDate

# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    LSOA_pre = pd.read_csv("./CrimeData/Processed/December_2012_to_march_2023.csv")
    pivot_table(LSOA_pre, index='LSOA code').to_csv("./CrimeData/Processed/pivot_LSOA.csv")