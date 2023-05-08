import pandas as pd
import datetime

from SmallDate import SmallDate


def get_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def parse_dataframe(path):
    df = get_dataset(path)
    df_burglaries = df[df["Crime type"] == "Burglary"]

    #df_burglaries.drop(columns=["Context"], inplace=True)
    return df_burglaries

def join_parsed_dataframes(date_initial, date_final_inclusive):
    df_all = None
    iteration_date = date_initial

    while iteration_date.smaller_than_or_equal_to(date_final_inclusive):
        year_str, month_str = str(iteration_date)[:4], str(iteration_date)[-2:]
        path = f'RawCrimeData/{year_str}-{month_str}-metropolitan-street.csv'
        iteration_df = parse_dataframe(path)

        if df_all is None:
            df_all = iteration_df
        else:
            df_all = pd.concat([df_all,iteration_df])
        print(str(iteration_date)+' has been completed')
        iteration_date.increase_month()
    return df_all

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    january_2022_to_march_2023 = join_parsed_dataframes(SmallDate(2022,1), SmallDate(2023,3))
    january_2022_to_march_2023.to_csv("./ProcessedData/january_2022_to_march_2023.csv")
    April_2020_to_march_2023 = join_parsed_dataframes(SmallDate(2020, 4), SmallDate(2023, 3))
    April_2020_to_march_2023.to_csv("./ProcessedData/April_2020_to_march_2023.csv")

