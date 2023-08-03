import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Load data into a DataFrame
df = pd.read_csv('modified_c4_epa_air_quality.csv')

# Display first 10 rows of the data.
print(df.head(10))

# Display number of rows, number of columns.
print(df.shape)

###############################################################################
############ Check for normal distribution using empirical rule ###############
###############################################################################

# Create a histogram to visualize distribution of aqi_log.
plt.figure(figsize=(10, 6))  # Optional: Set the figure size
plt.hist(df["aqi_log"], bins=10, edgecolor='black', color='skyblue')

# Set labels and title
plt.xlabel("AQI Log Value")
plt.ylabel("Frequency")
plt.title("Distribution of AQI Log")

# Display the histogram
plt.show()

# alternative way to create histogram
# df["aqi_log"].hist()

# There is a slight right skew, but it still appears to be a bell shape. 
# This shape suggests that the distribution of this data should be approximately normal.

# Use the empirical rule to observe the data, then test and verify that it is normally distributed.

# The empirical rule states that, for every normal distribution:

   # 68% of the data fall within 1 standard deviation of the mean
   # 95% of the data fall within 2 standard deviations of the mean
   # 99.7% of the data fall within 3 standard deviations of the mean

# Define variable for aqi_log mean.
mean_aqi_log = df["aqi_log"].mean()

# Define variable for aqi_log standard deviation.
std_aqi_log = df["aqi_log"].std()

print(std_aqi_log)

# To compute the actual percentage of the data that satisfies this criteria, 
# define the lower limit (for example, 1 standard deviation below the mean) 
# and the upper limit (for example, 1 standard deviation above the mean).

# Define variable for lower limit, 1 standard deviation below the mean.
lower_limit = mean_aqi_log - 1 * std_aqi_log

# Define variable for upper limit, 1 standard deviation above the mean.
upper_limit = mean_aqi_log + 1 * std_aqi_log

print(lower_limit, upper_limit)

# Display the actual percentage of data that falls within 1 standard deviation of the mean.
((df["aqi_log"] >= lower_limit) & (df["aqi_log"] <= upper_limit)).mean() * 100


# Define variable for lower limit, 2 standard deviations below the mean.
lower_limit_2 = mean_aqi_log - 2 * std_aqi_log

# Define variable for upper limit, 2 standard deviations below the mean.
upper_limit_2 = mean_aqi_log + 2 * std_aqi_log

# Display lower_limit, upper_limit.
print(lower_limit_2, upper_limit_2)

# Display the actual percentage of data that falls within 2 standard deviations of the mean.
((df["aqi_log"] >= lower_limit_2) & (df["aqi_log"] <= upper_limit_2)).mean() * 100

# Define variable for lower limit, 3 standard deviations below the mean.
lower_limit_3 = mean_aqi_log - 3 * std_aqi_log

# Define variable for upper limit, 3 standard deviations above the mean.
upper_limit_3 = mean_aqi_log + 3 * std_aqi_log

print(lower_limit_3, upper_limit_3)

# Display the actual percentage of data that falls within 3 standard deviations of the mean.
((df["aqi_log"] >= lower_limit_3) & (df["aqi_log"] <= upper_limit_3)).mean() * 100


# About 76.15% of the data falls within 1 standard deviation of the mean.
# About 95.77% of the data falls within 2 standard deviation of the mean.
# About 99.62% of the data falls within 3 standard deviations of the mean.

# The 95.77% is very close to 95%, and the 99.62% is very close to 99.7%. 
# The 76.15% is not as close to 68%, but relatively close. 
# Overall, from applying the empirical rule, the data appears to be not exactly normal, 
# but could be considered approximately normal. 


################################################
######## Using Z-score to find outliers ########
################################################

# Compute the z-score for every aqi_log value, and add a column named z_score in the data to store those results.
df["z_score"] = stats.zscore(df["aqi_log"])

# Display the first 5 rows to ensure that the new column was added.
print(df.head(10))

# Display data where `aqi_log` is above or below 3 standard deviations of the mean
df[(df["z_score"] > 3) | (df["z_score"] < -3)]

# The aqi_log for West Phoenix is slightly above 3 standard deviations of the mean. 
# This means that the air quality at that site is worse than the rest of the sites represented in the data.