import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Load the data.
data = pd.read_csv('marketing_sales_data.csv')

# Display the first five rows.
print(data.head())

# Create a boxplot with TV and Sales.
sns.boxplot(x = "TV", y = "Sales", data = data)
plt.show()
# There is considerable variation in Sales across the TV groups. 
# The significance of these differences can be tested with a one-way ANOVA.

# Create a boxplot with Influencer and Sales.
sns.boxplot(x = "Influencer", y = "Sales", data = data)
plt.show()
# There is some variation in Sales across the Influencer groups, but it may not be significant.

# Drop rows that contain missing data and update the DataFrame.
data = data.dropna(axis=0)

# Confirm the data contain no missing values.
data.isnull().sum(axis=0)

# Define the OLS formula.
ols_formula = 'Sales ~ C(TV)'

# Create an OLS model.
OLS = ols(formula = ols_formula, data = data)

# Fit the model.
model = OLS.fit()

# Save the results summary.
model_results = model.summary()

# Display the model results.
print(model_results)
# TV was selected as the preceding analysis showed a strong relationship between the TV promotion budget and the average Sales.
# Influencer was not selected because it did not show a strong relationship to Sales in the analysis.


# Calculate the residuals.
residuals = model.resid

# Create a 1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Create a histogram with the residuals.
sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")

# Set the title of the residual plot.
axes[0].set_title("Histogram of Residuals")

# Create a QQ plot of the residuals.
sm.qqplot(residuals, line='s',ax = axes[1])

# Set the title of the QQ plot.
axes[1].set_title("Normal QQ Plot")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()

# Show the plot.
plt.show()
# Is the normality assumption met?
# There is reasonable concern that the normality assumption is not met when TV is used as the independent variable predicting Sales. 
# The normal q-q forms an 'S' that deviates off the red diagonal line, which is not desired behavior. 

# Create a scatter plot with the fitted values from the model and the residuals.
fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x axis label
fig.set_xlabel("Fitted Values")

# Set the y axis label
fig.set_ylabel("Residuals")

# Set the title
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)

# Show the plot
plt.show()

# Is the constant variance (homoscedasticity) assumption met?
# The variance where there are fitted values is similarly distributed, validating that the constant variance assumption is met.

# Display the model results summary.
print(model_results)

# R-squared
# Using TV as the independent variable results in a linear regression model with 𝑅2=0.874. 
# In other words, the model explains 87.4% of the variation in Sales. 
# This makes the model an effective predictor of Sales. 

# coefficient estimates
# The default TV category for the model is High, because there are coefficients for the other two TV categories, Medium and Low. 
# According to the model, Sales with a Medium or Low TV category are lower on average than Sales with a High TV category. 
# For example, the model predicts that a Low TV promotion would be 208.813 (in millions of dollars) lower in Sales on average than a High TV promotion.

# The p-value for all coefficients is 0.000, meaning all coefficients are statistically significant at 𝑝=0.05. 
# The 95% confidence intervals for each coefficient should be reported when presenting results to stakeholders. 
# For instance, there is a 95% chance the interval [−215.353,−202.274] contains the true parameter of the slope of 𝛽𝑇𝑉𝐿𝑜𝑤,
# which is the estimated difference in promotion sales when a Low TV promotion is chosen instead of a High TV promotion.

# improvements
# Given how accurate TV was as a predictor, the model could be improved with a more granular view of the TV promotions, such as additional categories or the actual TV promotion budgets. 
# Further, additional variables, such as the location of the marketing campaign or the time of year, may increase model accuracy. 

# Create an one-way ANOVA table for the fit model to determine whether there is a statistically significant difference in Sales among groups. 
# The null hypothesis is that there is no difference in Sales based on the TV promotion budget.
# The alternative hypothesis is that there is a difference in Sales based on the TV promotion budget.
sm.stats.anova_lm(model, typ=2)

# The F-test statistic is 1971.46 and the p-value is 8.81∗10−256 (i.e., very small). 
# Because the p-value is less than 0.05, you would reject the null hypothesis that there is no difference in Sales based on the TV promotion budget.
# The results of the one-way ANOVA test indicate that you can reject the null hypothesis in favor of the alternative hypothesis. 
# There is a statistically significant difference in Sales among TV groups.

# Perform the Tukey's HSD post hoc test to compare if there is a significant difference between each pair of categories for TV.
tukey_oneway = pairwise_tukeyhsd(endog = data["Sales"], groups = data["TV"])

# Display the results
print(tukey_oneway.summary())

# The first row, which compares the High and Low TV groups, 
# indicates that you can reject the null hypothesis that there is no significant difference between the Sales of these two groups.
# You can also reject the null hypotheses for the two other pairwise comparisons that compare High to Medium and Low to Medium.
# Sales is not the same between any pair of TV groups. 

# stakeholder report: 
'''High TV promotion budgets result in significantly more sales than both medium and low TV promotion budgets. Medium TV promotion budgets result in significantly more sales than low TV promotion budgets.

Specifically, following are estimates for the difference between the mean sales resulting from different pairs of TV promotions, as determined by the Tukey's HSD test:

    Estimated difference between the mean sales resulting from High and Low TV promotions: $208.81 million (with 95% confidence that the exact value for this difference is between 200.99 and 216.64 million dollars).
    Estimated difference between the mean sales resulting from High and Medium TV promotions: $101.51 million (with 95% confidence that the exact value for this difference is between 93.69 and 109.32 million dollars).
    difference between the mean sales resulting from Medium and Low TV promotions: $107.31 million (with 95% confidence that the exact value for this difference is between 99.71 and 114.91 million dollars).

The linear regression model estimating Sales from TV had an R-squared of $0.871, making it a fairly accurate estimator. The model showed a statistically significant relationship between the TV promotion budget and Sales.

The results of the one-way ANOVA test indicate that the null hypothesis that there is no difference in Sales based on the TV promotion budget can be rejected. Through the ANOVA post hoc test, a significant difference between all pairs of TV promotions was found.

The difference in the distribution of sales across TV promotions was determined significant by both a one-way ANOVA test and a Tukey’s HSD test.''' 