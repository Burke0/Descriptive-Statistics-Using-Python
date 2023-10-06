import numpy as np
import pandas as pd
from scipy import stats

##########################################
######## Confidence Intervals ############
##########################################

education_districtwise = pd.read_csv('education_districtwise.csv')
education_districtwise = education_districtwise.dropna()

# n: smaple size 50
# replace: true to sample with replacement
# random_state: the same random number will generate the same results
sampled_data = education_districtwise.sample(n=50, replace=True, random_state = 31208)

# show 50 districts selected randomly from dataset
print(sampled_data)

# To construct a 95% confidence interval of the mean district literacy rate based on the sample data. 
# Use the four steps for constructing a confidence interval:

#1 Identify a sample statistic
#2 Choose a confidence level
#3 Find the margin of error
#4 Calculate the interval

sample_mean = sampled_data['OVERALL_LI'].mean()

# calulate std error: take std dev, divide by sqrt of sample
# shape[0] returns the number of rows in dataset(same as sample size)
estimated_standard_error = sampled_data['OVERALL_LI'].std() / np.sqrt(sampled_data.shape[0])

# construct the confidence interval

    # alpha(The confidence level): Enter 0.95 because you want to use a 95% confidence level
    # loc(The sample mean): Enter the variable sample_mean
    # scale(The sample standard error): Enter the variable estimated_standard_error
sample_confidence_interval = stats.norm.interval(0.95, loc=sample_mean, scale=estimated_standard_error)

print('95% confidence interval for the mean district literacy rate:', sample_confidence_interval)
