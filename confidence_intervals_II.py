import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# Load datset
aqi = pd.read_csv('c4_epa_air_quality.csv')

##########################################
###### Explore the `aqi` DataFrame #######
##########################################

print("Use describe() to summarize AQI")
print(aqi.describe(include='all'))

print("For a more thorough examination of observations by state use values_counts()")
print(aqi['state_name'].value_counts())

# Summarize the mean AQI for RRE states.
rre_states = ['California','Florida','Michigan','Ohio','Pennsylvania','Texas']
aqi_rre = aqi[aqi['state_name'].isin(rre_states)]
aqi_rre.groupby(['state_name']).agg({"aqi":"mean","state_name":"count"}) #alias as aqi_rre

##########################################################################
###### Construct a boxplot visualization for the AQI of these states #####
##########################################################################

plt.figure(figsize=(12, 6))
ax = sns.boxplot(x=aqi_rre["state_name"], y=aqi_rre["aqi"], palette="Set3")

# Customize the plot
ax.set_title("Distribution of AQI by State")
ax.set_xlabel("State")
ax.set_ylabel("AQI Value")

# Rotate the x-axis labels for better readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

# Display the plot
plt.tight_layout()
plt.show()

#######################################################################################
##### Construct a confidence interval for the RRE state with the highest mean AQI #####
#######################################################################################

# Find the mean aqi for your state.
aqi_ca = aqi[aqi['state_name']=='California']

sample_mean = aqi_ca['aqi'].mean()
print('The sample meain is: ', sample_mean)

# Input your confidence level.
confidence_level = 0.95

# Calculate your margin of error.

# Begin by identifying the z associated with your chosen confidence level.
z_value = 1.96

# Next, calculate your standard error.
standard_error = aqi_ca['aqi'].std() / np.sqrt(aqi_ca.shape[0])
print("standard error: ", standard_error)

# Lastly, use the preceding result to calculate your margin of error.
margin_of_error = standard_error * z_value
print("margin of error:", margin_of_error)

# Calculate your confidence interval (upper and lower limits).
upper_ci_limit = sample_mean + margin_of_error
lower_ci_limit = sample_mean - margin_of_error
print('The 95% confidence interval is:', (lower_ci_limit, upper_ci_limit))

# One-line Alternative
print('The 95% confidence interval is:', stats.norm.interval(confidence_level, loc=sample_mean, scale=standard_error))

# Based on the mean AQI for RRE states, California and Michigan were most likely to have experienced a mean AQI above 10.
# With California experiencing the highest sample mean AQI in the data, it appears to be the state most likely to be affected by the policy change. 