import pandas as pd
import numpy as np


def get_dataset(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = get_dataset("./data/2023-02-metropolitan-street.csv")
    df_burglaries = df[df["Crime type"] == "Burglary"]
