import pandas as pd
import numpy as np
from scipy import stats

# Task List: 
# 1) Is the mean AQI in Los Angeles County is statistically different from the rest of California?
# 2) Does New York have a lower AQI than Ohio?
# 3) A new policy will affect those states with a mean AQI of 10 or greater. Will Michigan be affected by this new policy?

aqi = pd.read_csv('c4_epa_air_quality.csv')

print("First 10 rows of data: ")
print(aqi.head(10))

print("AQI Summary: ")
print(aqi.describe(include='all'))

print("values_counts() by state name: ")
print(aqi['state_name'].value_counts())

# Task 1:  Is the mean AQI in Los Angeles County is statistically different from the rest of California?

# 𝐻0: There is no difference in the mean AQI between Los Angeles County and the rest of California.
# 𝐻𝐴: There is a difference in the mean AQI between Los Angeles County and the rest of California.

ca_la = aqi[aqi['county_name']=='Los Angeles']
ca_other = aqi[(aqi['state_name']=='California') & (aqi['county_name']!='Los Angeles')]

significance_level = 0.05

# Two sample T-test
print('Two sample t-test: ', stats.ttest_ind(a=ca_la['aqi'], b=ca_other['aqi'], equal_var=False))

# With a p-value (0.049) being less than 0.05 (as your significance level is 5%), reject the null hypothesis in favor of the alternative hypothesis.

# Task 2: Does New York have a lower AQI than Ohio?

# 𝐻0: The mean AQI of New York is greater than or equal to that of Ohio.
# 𝐻𝐴: The mean AQI of New York is below that of Ohio.

ny = aqi[aqi['state_name']=='New York']
ohio = aqi[aqi['state_name']=='Ohio']

# Two sample t-test 
tstat, pvalue = stats.ttest_ind(a=ny['aqi'], b=ohio['aqi'], alternative='less', equal_var=False)
print('Task 2 ')
print('tstat: ',tstat)
print('p value: ', pvalue)

# With a p-value (0.030) of less than 0.05 (as your significance level is 5%) and a t-statistic < 0 (-2.036), 
# reject the null hypothesis in favor of the alternative hypothesis.

# Task 3: A new policy will affect those states with a mean AQI of 10 or greater. Will Michigan be affected by this new policy?

# 𝐻0: The mean AQI of Michigan is less than or equal to 10.
# 𝐻𝐴: The mean AQI of Michigan is greater than 10.
michigan = aqi[aqi['state_name']=='Michigan']

# This is a comparison of one sample mean relative to a particular value in one direction. 
# This calls for a one-sample 𝑡-test.
tstat, pvalue = stats.ttest_1samp(michigan['aqi'], 10, alternative='greater')
print('Task 3 ')
print('t stat: ', tstat)
print('p value: ', pvalue)

# With a p-value (0.940) being greater than 0.05 (as your significance level is 5%)
# and a t-statistic < 0 (-1.74), 
# fail to reject the null hypothesis.

# Results:
# (At a 95% confidence level):
    # 1) The AQI in Los Angeles County was in fact different from the rest of California.
    # 2) New York has a lower AQI than Ohio based on the results.
    # 3) It can't be  concluded that the mean AQI is greater than 10. Thus, it is unlikely that Michigan would be affected by the new policy.
