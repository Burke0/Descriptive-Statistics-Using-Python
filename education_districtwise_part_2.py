import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm

education_districtwise = pd.read_csv('education_districtwise.csv')
education_districtwise = education_districtwise.dropna()

########################################
####### Simluate Random Sampling #######
########################################

# pandas sample function takes the following arguments
# n: Refers to the desired sample size
# replace: Indicates whether you are sampling with or without replacement
# random_state: Refers to the seed of the random number

# Sample from 50 districts, true for sampling with replacment, and some arbitrary number for the seed
sampled_data = education_districtwise.sample(n=50, replace=True, random_state=31208)
print(sampled_data)

# Compute the sample mean
estimate1 = sampled_data['OVERALL_LI'].mean()
print('The sample mean is: ', estimate1)

# Take a second sample and compute the mean
estimate2 = education_districtwise['OVERALL_LI'].sample(n=50, replace=True, random_state=56810).mean()
print('The second sample mean is:', estimate2)

################################################################################
######## Compute the mean of a sampling distribution with 10,000 samples #######
################################################################################

# Loop 10000 times and store the mean in estimate list each time without random state arguement so it automatically uses a different seed each time
estimate_list = []
for i in range(10000):
    estimate_list.append(education_districtwise['OVERALL_LI'].sample(n=50, replace=True).mean())
estimate_df = pd.DataFrame(data={'estimate': estimate_list})

# Compute the mean for your sampling distribution of 10,000 random samples. 
mean_sample_means = estimate_df['estimate'].mean()
print('Mean sample means: ',mean_sample_means)

# Compare this with the population mean of your complete dataset
population_mean = education_districtwise['OVERALL_LI'].mean()
print('population mean: ', population_mean)

# The mean of the sampling distribution is essentially identical to the population mean, which is also about 73.4%!

# This plot shows the Central Limit Theorem in action, 
# visualizing how sample means from random samples tend to converge around the population mean as the number of samples increases.
plt.hist(estimate_df['estimate'], bins=25, density=True, alpha=0.4, label = "histogram of sample means of 10000 random samples")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100) # generate a grid of 100 values from xmin to xmax.
p = stats.norm.pdf(x, mean_sample_means, stats.tstd(estimate_df['estimate']))
plt.plot(x, p,'k', linewidth=2, label = 'normal curve from central limit theorem')
plt.axvline(x=population_mean, color='g', linestyle = 'solid', label = 'population mean')
plt.axvline(x=estimate1, color='r', linestyle = '--', label = 'sample mean of the first random sample')
plt.axvline(x=mean_sample_means, color='b', linestyle = ':', label = 'mean of sample means of 10000 random samples')
plt.title("Sampling distribution of sample mean")
plt.xlabel('sample mean')
plt.ylabel('density')
plt.legend(bbox_to_anchor=(1.04,1))
plt.show()