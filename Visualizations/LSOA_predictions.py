import pandas as pd
import numpy as py
import csv
import plotly.express as px
import json

Location = 'LSOA'
ward_predicitions = pd.read_csv(r'../CrimeData/Processed/model_prediction_LSOA.csv')
print(ward_predicitions)

with open("../GeographicalData/Raw/lsoa.geojson", "r") as geojsonfile:
    geojson = json.load(geojsonfile)
featureidkey = "properties.LSOA11CD"

fig = px.choropleth_mapbox(ward_predicitions, geojson=geojson, color='Crime Likelihood',
                           locations=Location, featureidkey=featureidkey,
                           center={"lat": 51.509865, "lon": -0.118092},
                           mapbox_style="carto-positron", zoom=9,
                           labels=[Location, 'Crime Likelihood'])
fig.show()
