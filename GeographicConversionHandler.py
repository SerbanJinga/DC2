import numpy as np
import pandas as pd


def getConversionDf(year=2011):
    """
    Crime data is in the format of 2011 despite the year of the crime, always use it unless you know what you are doing
    """
    path = None
    assert year in [2011, 2021], 'Year must be 2011 or 2021'
    if year == 2011:
        return pd.read_excel('GeographicalData/Raw/LSOA11_WD21_LAD21_EW_LU_V2.xlsx')
    if year == 2021:
        return pd.read_csv(
            'GeographicalData/Raw/LSOA_(2021)_to_Ward_to_Lower_Tier_Local_Authority_(May_2022)_Lookup_for_England_and_Wales.csv')
class GeographicConversionHandler:

    def __init__(self):
        self.year = 2011 #"DO NOT CHANGE"
        self.table = getConversionDf(self.year)

    def LSOA_to_ward(self):
        Ward_from_LSOA = {}
        for index, row in self.table.iterrows():
            Ward_from_LSOA[row['LSOA11CD']] = row['WD21NM']
        return Ward_from_LSOA

    def Barnet_Data(self):
        df = self.table
        if self.year == 2011:
            return df[df['LAD21NM']=='Barnet']
        if self.year == 2021:
            return df[df['LTLA22NM']=='Barnet']

    def Barnet_LSOA(self):
        if self.year == 2011:
            return self.Barnet_Data()['LSOA11CD']
        if self.year == 2021:
            return self.Barnet_Data()['LSOA21CD']

    def all_LSOA(self):
        if self.year == 2011:
            return set(self.table['LSOA11CD'])
        if self.year == 2021:
            return set(self.table['LSOA21CD'])

    def geo_conv_matrix(self, wards, LSOAs):
        convesion_matrix = np.zeros((len(wards), len(LSOAs)))
        geoHandler_dict = self.LSOA_to_ward()
        for i in range(len(wards)):
            for j in range(len(LSOAs)):
                if geoHandler_dict[LSOAs[j]] == wards[i]:
                    convesion_matrix[i, j] = 1
        return convesion_matrix