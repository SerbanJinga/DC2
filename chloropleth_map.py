import pandas as pd
import numpy as py
import csv
import plotly.express as px
import json



dataset = pd.read_csv('ProcessedData/january_2022_to_march_2023.csv')
with open('RawCrimeData/lsoa.geojson', 'r') as geojsonfile:
    geojson = json.load(geojsonfile)


grouped_data = dataset.groupby(['LSOA code'])['Crime ID'].agg('count')
print(grouped_data)
dictionary = grouped_data.to_dict()
print(dictionary)

dataset['count'] = dataset['LSOA code'].apply(lambda x: dictionary[x])
print(dataset)

fig = px.choropleth_mapbox(dataset, geojson=geojson, color='count',
                           locations="LSOA code", featureidkey="properties.LSOA11CD",
                           center={"lat": 51.509865, "lon": -0.118092},
                           mapbox_style="carto-positron", zoom=9,
                           labels= ['LSOA name','count'])
fig.show()
