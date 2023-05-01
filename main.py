import pandas as pd
import numpy as np
import time 


def get_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def parse_dataframe(path):
    df = get_dataset(path)
    df_burglaries = df[df["Crime type"] == "Burglary"].copy() 
    df_burglaries_Barnet_february = pd.DataFrame(columns=df_burglaries.columns)
    for index in range(len(df_burglaries)):
        # print(df_burglaries[index])
        x = df_burglaries.iloc[index]["LSOA code"]
        if isinstance(x, float):
            continue
        else:
            if int(x[2:]) >= 1000001 and int(x[2:]) <= 1004747:
                df_burglaries_Barnet_february.loc[index] = (df_burglaries.iloc[index])

    df_burglaries_Barnet_february.drop(columns=["Context"], inplace=True)
    return df_burglaries_Barnet_february


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_burglaries_Barnet_february = parse_dataframe("./data/2023-02-metropolitan-street.csv")
    df_burglaries_Barnet_march = parse_dataframe("./data/2023-02-metropolitan-street.csv")
    january = parse_dataframe("./data/2023-01-metropolitan-street.csv")
    december = parse_dataframe("./data/2022-12-metropolitan-street.csv")
    november = parse_dataframe("./data/2022-11-metropolitan-street.csv")
    october = parse_dataframe("./data/2022-10-metropolitan-street.csv")
    september = parse_dataframe("./data/2022-09-metropolitan-street.csv")
    august = parse_dataframe("./data/2022-08-metropolitan-street.csv")
    july = parse_dataframe("./data/2022-07-metropolitan-street.csv")
    june = parse_dataframe("./data/2022-06-metropolitan-street.csv")
    may = parse_dataframe("./data/2022-05-metropolitan-street.csv")
    april = parse_dataframe("./data/2022-04-metropolitan-street.csv")
    march = parse_dataframe("./data/2022-03-metropolitan-street.csv")
    february = parse_dataframe("./data/2022-02-metropolitan-street.csv")
    january = parse_dataframe("./data/2022-01-metropolitan-street.csv")

    df_all = pd.concat([january, february, march, april, may, june, july, august, september, october, november, december, january, df_burglaries_Barnet_february, df_burglaries_Barnet_march])
    df_all.to_csv("./data/january_2022_to_march_2023.csv")
