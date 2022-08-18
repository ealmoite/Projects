import pandas as pd
from pyproj import Proj
import numpy as np
import math

# This code converts UTM coordinates from the crime dataset into latitude and longitude.
# Then, it iterates through the crime dataset and chooses the dissemination area closest to the crime.

# Receives X_test values and finds closest cluster for each.
def predictCluster(centroids, X_test, clusterLabels):
    bestClusterList = []

    for i in range(0, len(X_test)):
        smallestDistance = None
        bestCluster = None
        print(i, len(X_test))

        # Compare each X value proximity with all centroids.
        for row in range(centroids.shape[0]):
            distance = 0

            # Get absolute distance between centroid and X.
            for col in range(centroids.shape[1]):
                distance += \
                    math.sqrt((centroids[row][col] - X_test.iloc[i][col]) ** 2)

            # Initialize bestCluster and smallestDistance during first iteration.
            # OR re-assign bestCluster if smaller distance to centroid found.
            if (bestCluster == None or distance < smallestDistance):
                bestCluster = clusterLabels[row]
                smallestDistance = distance

        bestClusterList.append(bestCluster)

    return bestClusterList


PATH = "C:\\datasets\\"
CSV_DATA = "crimedata.csv"
LATLONG_DATA = "C:\\datasets\\crime\\dis_area_lat_long.csv"

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

df = pd.read_csv(PATH + CSV_DATA)
df = df.dropna(how='any', axis=0)

# This converts UTM coordinates into latitude and longitude
myProj = Proj("+proj=utm +zone=10 +north +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
lon, lat = myProj(df['X'].values, df['Y'].values, inverse=True)
UTMx, UTMy = myProj(lon, lat)
df['UTMx'] = UTMx
df['UTMy'] = UTMy
df['lon'] = lon
df['lat'] = lat
df = df[(df['YEAR'] == 2016)]

# New dataframe with latitude and longitude
print(df)

# This dataframe shows the latitude and longitude of dissemination areas in Vancouver
dis_df = pd.read_csv(LATLONG_DATA)
dis_df = dis_df[[' DAuid/ADidu', ' DArplat/Adlat', ' DArplong/ADlong']]
print(dis_df.head(10))

geom = dis_df[[' DArplat/Adlat', ' DArplong/ADlong']]
print(geom)

# Assign centroids
centroids = np.zeros((len(dis_df), 2))
for row in range(0, len(dis_df)):
    for col in range(0, 2):
        centroids[row][col] = geom.iloc[row, col]
print(centroids)

clusterLabels = []
for i in range(0, len(dis_df)):
    clusterLabels.append(dis_df.loc[i, " DAuid/ADidu"])
print(clusterLabels)

X_test = df[['lon', 'lat']]
predictions = predictCluster(centroids, X_test, clusterLabels)
print('**********Predictions************')
print(predictions)

df['dis_area'] = 0
for i in range(0, len(predictions)):
    df.at[i, 'dis_area'] = clusterLabels[i]
    print(i)
print(df)