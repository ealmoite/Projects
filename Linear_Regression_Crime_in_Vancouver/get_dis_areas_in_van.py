import numpy as np
import pandas as pd

# This script selects all the dissemination areas from the 2016 census that are in Greater Vancouver

# This is the 2016 census
PATH = "C:\\datasets\\crime\\"
CSV_DATA = "census2016.csv"
df = pd.read_csv(PATH + CSV_DATA)

# Show all columns.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# This creates a list of all dissemination areas in Greater Vancouver
dfVan = pd.read_csv("C:\\datasets\\crime\\dis_area_lat_long.csv")
dArea = dfVan[' DAuid/ADidu'].tolist()

# Filtering the 2016 census to include only Greater Vancouver areas
df2 = df[df['GEO_CODE (POR)'].isin(dArea)]

# Remove columns with 'xx'
# df2 = df2[(df2['Dim: Sex (3): Member ID: [1]: Total - Sex']!='xx')]
# df2.sort_index()
# df2.to_csv(path_or_buf="C:\\datasets\\crime\\census_van_area.csv")

# Pivoting the table
table = pd.pivot_table(df2
                       , values=["Dim: Sex (3): Member ID: [1]: Total - Sex"]
                       , index=['GEO_CODE (POR)']
                       , columns=["Member ID: Profile of Dissemination Areas (2247)", "DIM: Profile of Dissemination Areas (2247)"]
                       , aggfunc=np.sum
                       , sort=False
                   )

# table.to_csv(path_or_buf="C:\\datasets\\crime\\census_with_dis_area.csv")
