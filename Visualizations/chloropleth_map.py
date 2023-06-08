import pandas as pd
import numpy as py
import csv
import plotly.express as px
import json

dataset = pd.read_csv('../CrimeData/Processed/December_2012_to_march_2023.csv')
print(dataset.columns)
with open('../GeographicalData/Raw/GeoWards.geojson', 'r') as geojsonfile:
    geojson = json.load(geojsonfile)

dataset.drop(dataset[dataset['Ward'].isnull()].index, inplace=True)

dataset['Month'] = pd.to_datetime(dataset['Month'])
next_month = dataset[dataset['Month'].dt.strftime('%Y-%m') == '2018-06']
print(next_month[['Month', 'Ward']])

grouped_data = next_month.groupby(['Ward'])['Crime ID'].agg('count')
grouped = pd.DataFrame(grouped_data).reset_index()
grouped.columns = ['Ward', 'Count']
grouped['percentage'] = (grouped['Count']/grouped['Count'].sum())*100


def largest_remainder_round(df, column_name):
    total = df[column_name].sum()
    df['rounded'] = df[column_name] // 1  # Floor division
    remainder = int(total - df['rounded'].sum())  # Convert remainder to integer
    rounded_values = df['rounded'].values.tolist()
    sorted_indices = sorted(range(len(rounded_values)), key=lambda i: -(rounded_values[i] % 1))

    for i in range(remainder):
        rounded_values[sorted_indices[i]] += 1
    print(df)
    df['rounded'] = rounded_values

    return df
grouped= largest_remainder_round(grouped,'percentage')

print(grouped['rounded'].sum())



fig = px.choropleth_mapbox(grouped, geojson=geojson, color='rounded',
                           locations="Ward", featureidkey="properties.WD13NM",
                           center={"lat": 51.509865, "lon": -0.118092},
                           mapbox_style="carto-positron", zoom=9,
                           labels= ['Ward','count'])
fig.show()


# MODEL PREDICTION VISUALIZATION







