import pandas as pd

from GeographicConversionHandler import GeographicConversionHandler

if __name__ == "__main__":
    df = pd.read_csv("CrimeData/Processed/january_2022_to_march_2023.csv")
    LSOA = df['LSOA code'].unique()
    geoHand = GeographicConversionHandler()
    Wards = df['LSOA code'].map(geoHand.LSOA_to_ward()).unique()
    print(len(Wards)) # len of the dataset is now 150000