import pandas as pd
from scipy import stats

################################
###### Hypothesis Testing ######
################################

education_districtwise = pd.read_csv("education_districtwise.csv")
education_districtwise = education_districtwise.dropna()

state21 = education_districtwise[education_districtwise['STATNAME'] == "STATE21"]
state28 = education_districtwise[education_districtwise['STATNAME'] == "STATE28"]

# Simulate random sampling(20 districts from each state)
    # n: sample size 20
    # replace: true to sample with replacement
    # random_state - arbitrary number for random seed

sampled_state21 = state21.sample(n=20, replace = True, random_state=13490)
sampled_state28 = state28.sample(n=20, replace = True, random_state=39103)

# Compute sample means
mean21 = sampled_state21['OVERALL_LI'].mean()
mean28 = sampled_state28['OVERALL_LI'].mean()
print('State 21 mean: ', mean21)
print('State 28 mean: ', mean28)

# The difference is about 6.2%
print('The difference between the mean district literacy rates of state 21 and 28 is: ', (mean21-mean28))

#####################################
##### Conduct a hypothesis test #####
#####################################

# A two-sample t-test is the standard approach for comparing the means of two independent samples.
    # Steps:
    # 1) State the null hypothesis and the alternative hypothesis
    # 2) Choose a significance level
    # 3) Find the P-value
    # 4) Reject or fail to reject the null hypothesis
    
# 𝐻0: There is no difference in the mean district literacy rates between STATE21 and STATE28.
# 𝐻𝐴: There is a difference in the mean district literacy rates between STATE21 and STATE28.

# We will use a sig level of 0.05 or 5%

# For a two-sample 𝑡-test, you can use scipy.stats.ttest_ind() to compute your p-value. This function includes the following arguments:

    # a: Observations from the first sample
    # b: Observations from the second sample
    # equal_var: A boolean, or true/false statement, which indicates whether the population variance of the two samples is assumed to be equal. In our example, you don’t have access to data for the entire population, so you don’t want to assume anything about the variance. To avoid making a wrong assumption, set this argument to False.

print('p-value: ', stats.ttest_ind(a=sampled_state21['OVERALL_LI'], b=sampled_state28['OVERALL_LI'], equal_var=False))

# The p value is about 0.0064: 
# This means there is only a 0.64% probability that the absolute difference between the two mean district literacy rates would be 6.2% or greater if the null hypothesis were true.
# Therefore, it is highly unlikely that the difference in the two means is due to chance.

# Since, p-value is less than the significance level of 5% we reject the null hypothesis.
# Thus, there is a statistically significant difference between the mean literacy rates of the two states.
