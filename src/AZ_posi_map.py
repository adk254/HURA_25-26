import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import utm
import numpy as np

# path to .shp file
shapefile_path = "AZ_subdvsns\\cb_2022_04_cousub_500k.shp"

# path to .csv file
csv_path = "SiteMaster2023_region.csv"

# read data from the csv file - contains utm and positive cases
sites_data = pd.read_csv(csv_path)

# colours
colours = ['#FB0650', '#1E88E5', '#07FF81', '#004D40']

# convert utm to lat long
latitudes = []
longitudes = []

for index, row in sites_data.iterrows():
    try:
        # check if coordinates are valid
        if pd.notna(row['m E']) and pd.notna(row['m N']):
            easting = float(row['m E'])
            northing = float(row['m N'])
            zone_number = int(row['UTM Zone'].rstrip('S').rstrip('R'))
            zone_string = str(row['UTM Zone'])
            if 'S' in zone_string:
                zone_letter = 'S'
            elif 'R' in zone_string:
                zone_letter = 'R'
            else:
                zone_letter = 'N'

            # convert utm t lat long
            lat, long = utm.to_latlon(easting, northing, zone_number, zone_letter)

            latitudes.append(lat)
            longitudes.append(long)
        else:
            latitudes.append(np.nan)
            longitudes.append(np.nan)
    except Exception as e:
        print(f"Error on row {index}: {e}")
        latitudes.append(np.nan)
        longitudes.append(np.nan)

# put lat long into dataframe
sites_data['Latitude'] = latitudes
sites_data['Longitude'] = longitudes

# filter for positive cases
positive_sites = sites_data[sites_data['Test Result'] == 1].dropna(
    subset=['Latitude', 'Longitude'])

# create GeoDataFrame using pos cases
geometry = gpd.points_from_xy(positive_sites['Longitude'], positive_sites['Latitude'])
positive_gdf = gpd.GeoDataFrame(positive_sites, geometry = geometry, crs="EPSG:4326")

# load the country subdivisons shapefile
az_subdivisions = gpd.read_file(shapefile_path)

# create figure and axis
fig, ax = plt.subplots(figsize = (12, 10))

# plot the county subdivisions
az_subdivisions.plot(ax=ax, color='lightgrey', edgecolor='black', linewidth=0.5)

# plot positive cases by region
for region_value, colour in zip([1, 2, 3, 4], colours):
    # filter for positive cases in this region
    region_sites = positive_gdf[positive_gdf['Region'] == region_value]

    if not region_sites.empty:
        region_sites.plot(
            ax = ax,
            color = colour,
            markersize = 75, marker = 'o',
            alpha = 0.7,
            label = f'Region {region_value}'
        )

plt.legend()

# set the title
#plt.title('Figure 1. Positive ATV Cases in Arizona by Region', fontsize=16)

# show plot
plt.tight_layout(pad=1)
plt.margins(0)
plt.axis('off')

plt.show()
