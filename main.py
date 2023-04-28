import pandas as pd
import numpy as np


def get_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = get_dataset("./data/2023-02-metropolitan-street.csv")
    df_burglaries = df[df["Crime type"] == "Burglary"].copy()
    # print(df_burglaries["LSOA code"].unique())
    # print(df_burglaries["LSOA name"].unique())
    df_burglaries_Barnet = pd.DataFrame(columns=df_burglaries.columns)
    for index in range(len(df_burglaries)):
        # print(df_burglaries[index])
        x = df_burglaries.iloc[index]["LSOA code"]
        if isinstance(x, float):
            continue
        else:
            if int(x[2:]) >= 1000001 and int(x[2:]) <= 1004747:
                df_burglaries_Barnet.loc[index] = (df_burglaries.iloc[index])

    df_burglaries_Barnet.drop(columns=["Context"], inplace=True)
    print(df_burglaries_Barnet.head(20))
