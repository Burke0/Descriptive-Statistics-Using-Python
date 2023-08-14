import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

# Load data into dataframe
epa_data = pd.read_csv('c4_epa_air_quality.csv', index_col = 0)

# Output first 10 rows of data
print(epa_data.head(10))

# Generate a table of some descriptive statistics about the data for all columns
print(epa_data.describe(include = 'all'))

# Find the mean of the aqi column
population_mean = epa_data['aqi'].mean()
print('population mean: ', population_mean)

# 50 random samples with replacement
sampled_data = epa_data.sample(n=50, replace=True, random_state=42)

# output first 10 rows of sample data
print(sampled_data.head(10))

# Find the mean of sample data
sample_mean = sampled_data['aqi'].mean()
print('sample data mean: ', sample_mean)

############################################################
############# Apply Central Limit Theorem ##################
############################################################

# According to the central limit theorem, 
# the mean of a sampling distribution should be roughly equal to the population mean

estimate_list = []
for i in range(10000):
    estimate_list.append(epa_data['aqi'].sample(n=50,replace=True).mean())

# Create a new dataframe from the list of 10,000 estimates
estimate_df = pd.DataFrame(data={'estimate': estimate_list})
print(estimate_df)

# Find the mean() of the sampling distribution
mean_sample_means = estimate_df['estimate'].mean()
print('mean sample means: ', mean_sample_means)

# Show the distribution of the estimates using a histogram
estimate_df['estimate'].hist()
plt.title('Air Quality Index Sampling Distribution')
plt.show()

############################################################
############# Calculate the standard error #################
############################################################

standard_error = sampled_data['aqi'].std() / np.sqrt(len(sampled_data))
print('standard error: ', standard_error)

################################################################################
### Visualize the relationship between the sampling and normal distributions ###
################################################################################

# Generate a grid of 100 values from xmin to xmax.
plt.figure(figsize=(8,5))
plt.hist(estimate_df['estimate'], bins=25, density=True, alpha=0.4, label = "histogram of sample means of 10000 random samples")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100) # generate a grid of 100 values from xmin to xmax.
p = stats.norm.pdf(x, population_mean, standard_error)
plt.plot(x, p, 'k', linewidth=2, label = 'normal curve from central limit theorem')
plt.axvline(x=population_mean, color='m', linestyle = 'solid', label = 'population mean')
plt.axvline(x=sample_mean, color='r', linestyle = '--', label = 'sample mean of the first random sample')
plt.axvline(x=mean_sample_means, color='b', linestyle = ':', label = 'mean of sample means of 10000 random samples')
plt.title("AQI sampling distribution of sample mean")
plt.xlabel('sample mean')
plt.ylabel('density')
plt.legend(bbox_to_anchor=(1.04,1))
plt.show()