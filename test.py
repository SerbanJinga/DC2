import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./data/january_2022_to_march_2023.csv")
    print(df.head(20))