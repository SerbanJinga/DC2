import pandas as pd
import numpy as py
import csv
import plotly.express as px
import json

import shapely
import topojson
import geopandas as gpd

ward_predictions = pd.read_csv(r'../CrimeData/Processed/model_prediction_Ward.csv')
ward_predictions['Crime_Likelihood'] = ward_predictions['Crime_Likelihood'].astype(float)
print(ward_predictions)


def convert_topojson_to_geojson(input_file_path, output_file_path):
    gdf = gpd.read_file(input_file_path)
    gdf.to_file(output_file_path, driver='GeoJSON')
input_file_path = '../GeographicalData/Raw/Wards.json'
output_file_path = '../GeographicalData/Raw/GeoWards.geojson'
convert_topojson_to_geojson(input_file_path, output_file_path)




with open("../GeographicalData/Raw/GeoWards.geojson", "r") as geojsonfile:
    geojson = json.load(geojsonfile)
featureidkey ='properties.id'

fig = px.choropleth_mapbox(ward_predictions,
                           geojson=geojson,
                           color='rounded',
                           locations='Ward',
                           featureidkey='properties.WD13NM',
                           center={"lat": 51.509865, "lon": -0.118092},
                           mapbox_style="carto-positron",
                           zoom=9,
                           hover_data=['Ward', 'Crime_Likelihood'],
                           labels={'Ward': 'Ward Name'}
                           )



fig.show()
