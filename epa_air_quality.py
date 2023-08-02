import pandas as pd
import numpy as np

epa_data = pd.read_csv('c4_epa_air_quality.csv', index_col = 0)

# display first 10 rows of data
print(epa_data.head(10))

# get descriptive stats
print(epa_data.describe())

# get descriptive stats about the states in the data
epa_data['state_name'].describe()

###########################################################
####### How to individually calculate statistics ##########
###########################################################

# output the mean value from the aqi(EPA's Air Quality Index) column
print(np.mean(epa_data['aqi']))

#  output the median value from the aqi
print(np.median(epa_data['aqi']))

# output the minimum value from the aqi column
print(np.min(epa_data['aqi']))

# output the maximum value from the aqi column
print(np.max(epa_data['aqi']))

# Compute the standard deviation for the aqi column
# By default, the numpy library uses 0 as the Delta Degrees of Freedom, 
# while pandas library uses 1. 
# To get the same value for standard deviation using either library, 
# specify the ddof parameter to 1 when calculating standard deviation
print(np.std(epa_data['aqi'], ddof=1))

# In this data set 75% of AQI values are below 9, which is considered good air quality
# Funding should be allocated for further investigation of the less healthy regions in order to learn how to improve the conditions. 






