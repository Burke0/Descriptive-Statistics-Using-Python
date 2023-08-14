import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('education_districtwise.csv')
 
# print first 10 rows of data
print(df.head(10))

#########################################################
###### Use describe() to compute descriptive stats ######
#########################################################

# Data professionals use the describe() function as a convenient way to calculate many key stats all at once. For a numeric column, describe() gives you the following output:

# count: Number of non-NA/null observations
# mean: The arithmetic average
# std: The standard deviation
# min: The smallest (minimum) value
# 25%: The first quartile (25th percentile)
# 50%: The median (50th percentile)
# 75%: The third quartile (75th percentile)
# max: The largest (maximum) value

# The OVERALL_LI is literacy rate for each district in the nation
print(df['OVERALL_LI'].describe())

# Desccribe also works for categorical data and outputs the following

# count: Number of non-NA/null observations
# unique: Number of unique values
# top: The most common value(mode)
# freq: The frequency of the most common value

print(df['STATNAME'].describe())

##########################################################
######### Use Min and Max to commpute range ##############
##########################################################

range_overall_literacy_rate= df['OVERALL_LI'].max() - df['OVERALL_LI'].min()
print("the range in literacy rates for all districts is ", range_overall_literacy_rate, "%")