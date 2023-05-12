import pandas as pd
import datetime

from GeographicConversionHandler import GeographicConversionHandler
from SmallDate import SmallDate

# Outside loop so the program runs fast
geoConvHandler = GeographicConversionHandler()
Barnet_areas = geoConvHandler.Barnet_LSOA()
Ward_dict = geoConvHandler.LSOA_to_ward()

def get_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def parse_dataframe(path):
    df = get_dataset(path)

    # First we drop the columns with null LSOA, as they are few and not useful for out anaylisis
    df.drop(df[df['LSOA code'].isnull()].index, inplace=True)
    # Second begining we check that the labels are valid
    assert not set(df['LSOA code'])-geoConvHandler.all_LSOA(), 'There are codes not registered in our geoHandler'

    # Third we filter all those in Barnet
    df = df[df['LSOA code'].isin(Barnet_areas)]

    # Fourth we take only the burglary ones
    #df = df[df["Crime type"] == "Burglary"]

    # Fifth we drop useless columns
    df.drop(columns=["Context"], inplace=True) # Nan column
    df.drop(columns=['Reported by','Falls within',"Longitude", "Latitude", "Location", "LSOA name", "Last outcome category"], inplace=True) # useless

    # Sixth, we add a Ward column
    df['Ward'] = df['LSOA code'].map(Ward_dict)

    return df

def join_parsed_dataframes(date_initial, date_final_inclusive):
    df_all = None
    iteration_date = date_initial

    while iteration_date.smaller_than_or_equal_to(date_final_inclusive):
        year_str, month_str = str(iteration_date)[:4], str(iteration_date)[-2:]
        path = f'CrimeData/Raw/{year_str}-{month_str}-metropolitan-street.csv'
        iteration_df = parse_dataframe(path)

        if df_all is None:
            df_all = iteration_df
        else:
            df_all = pd.concat([df_all,iteration_df])
        print(str(iteration_date)+' has been completed')
        iteration_date.increase_month()
    return df_all

# Press the green button in the gutter to run the script.
def pre_process_raw_data():
    january_2022_to_march_2023 = join_parsed_dataframes(SmallDate(2022,1), SmallDate(2023,3))
    january_2022_to_march_2023.to_csv("./CrimeData/Processed/january_2022_to_march_2023.csv")
    April_2020_to_march_2023 = join_parsed_dataframes(SmallDate(2020, 4), SmallDate(2023, 3))
    April_2020_to_march_2023.to_csv("./CrimeData/Processed/April_2020_to_march_2023.csv")


def pivot_table(df, index = 'Ward'):
    index = 'Ward'
    useful_columns_for_pivot = ['Crime ID', "Month", index]
    Usefuldata = df[useful_columns_for_pivot]
    pivot_table = Usefuldata.pivot_table(index=index, columns='Month', values='Crime ID', aggfunc='count')
    return pivot_table