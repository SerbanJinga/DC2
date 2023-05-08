import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("ProcessedData/january_2022_to_march_2023.csv")
    print(len(df)) # len of the dataset is now 150000