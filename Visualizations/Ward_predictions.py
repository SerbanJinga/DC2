import pandas as pd
import numpy as py
import csv
import plotly.express as px
import json

import shapely
import topojson
import geopandas as gpd

# ward_id = {'Brunswick Park' : 'E05000043',
#            'Burnt Oak' : 'E05000044',
#            'Childs Hill': "E05000045",
#            "Colindale" :"E05000046",
#            'Coppetts' : "E05000047",
#            "East Barnet": "E05000048",
#            "East Finchley": "E05000049",
#            "Edgware": "E05000050",
#            "Finchley Church End": "E05000051",
#            "Garden Suburb": "E05000052",
#            "Golders Green": "E05000053",
#            "Hale": "E05000054",
#             "Hendon": "E05000055",
#             "High Barnet": "E05000056",
#             "Mill Hill": "E05000057",
#             "Oakleigh": "E05000058",
#             "Totteridge": "E05000059",
#             "Underhill": "E05000060",
#             "West Finchley": "E05000061",
#             "West Hendon": "E05000062",
#             "Woodhouse": "E05000063",
# }

ward_predictions = pd.read_csv(r'../CrimeData/Processed/model_prediction_Ward.csv')
# ward_predictions.drop('Unnamed: 0')
print(ward_predictions)
# ward_predictions['id'] = ward_predictions['Ward'].apply(lambda x: ward_id[x])
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
                           color='Crime_Likelihood',
                           locations='Ward',
                           featureidkey='properties.WD13NM',
                           center={"lat": 51.509865, "lon": -0.118092},
                           mapbox_style="carto-positron",
                           zoom=9,
                           hover_data=['Ward', 'Crime_Likelihood'],
                           labels={'Ward': 'Ward Name'}
                           )



fig.show()
