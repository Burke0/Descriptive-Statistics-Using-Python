import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Load the data.
data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')

############################
##### Data exploration #####
############################

# Display the first five rows.
print(data.head())

# Display the shape of the data as a tuple (rows, columns).
print(data.shape)

# Generate descriptive statistics about TV, Radio, and Social_Media.
print(data[['TV','Radio','Social_Media']].describe())

# Calculate the average missing rate in the sales column.
missing_sales = data.Sales.isna().mean()

# Convert the missing_sales from a decimal to a percentage and round to 2 decimal places.
missing_sales = round(missing_sales*100, 2)

# Display the results (missing_sales must be converted to a string to be concatenated in the print statement).
print('Percentage of promotions missing Sales: ' +  str(missing_sales) + '%')

# Subset the data to include rows where Sales is present.
data = data.dropna(subset = ['Sales'], axis = 0)

# Create a histogram of the Sales.
fig = sns.histplot(data['Sales'])
fig.set_title('Distribution of Sales')
plt.show()
plt.clf()
# This shows sales are equally distributed between 250 and 350 million. 

############################
###### Model Building ######
############################

# Create a pairplot of the data.
sns.pairplot(data)
plt.show()
plt.clf()

# Define the OLS formula.
ols_formula = 'Sales ~ TV'

# Create an OLS model.
OLS = ols(formula = ols_formula, data = data)

# Fit the model.
model = OLS.fit()

# Save the results summary.
model_results = model.summary()

# Display the model results.
print(model_results)

#####################################
###### Check model assumptions ######
#####################################
# To justify using simple linear regression, check that the four linear regression assumptions are not violated. These assumptions are:

   # Linearity
   # Independent Observations
   # Normality
   # Homoscedasticity
   
# Model assumption: Linearity   
# Create a scatterplot comparing X and Sales (Y). 
# There is a clear linear relationship between TV and Sales, meeting the linearity assumption.
sns.scatterplot(x = data['TV'], y = data['Sales'])
plt.show()
plt.clf()
# Model assumption: Independence
# The independent observation assumption states that each observation in the dataset is independent.
# As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# Model assumption: Normality
    #  Histogram of the residuals
    #  Q-Q plot of the residuals

# Calculate the residuals.
residuals = model.resid

# Create a 1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Create a histogram with the residuals .
sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")

# Set the title of the residual plot.
axes[0].set_title("Histogram of Residuals")

# Create a Q-Q plot of the residuals.
sm.qqplot(residuals, line='s',ax = axes[1])

# Set the title of the Q-Q plot.
axes[1].set_title("Normal Q-Q plot")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()
plt.show()
plt.clf()

# The histogram of the residuals are approximately normally distributed, which supports that the normality assumption is met for this model.
# The residuals in the Q-Q plot form a straight line, further supporting that the normality assumption is met.

# Model assumption: Homoscedasticity

# Create a scatterplot with the fitted values from the model and the residuals.
fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x-axis label.
fig.set_xlabel("Fitted Values")

# Set the y-axis label.
fig.set_ylabel("Residuals")

# Set the title.
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.
fig.axhline(0)
plt.show()
plt.clf()

# The variance of the residuals is consistant across all 𝑋. Thus, the assumption of homoscedasticity is met.

# All assumptions are met, so the model results caan be interpreted accurately
print(model_results)
# Key takeaways

# When TV is used as the independent variable, it has a p-value of 0.000
# and a 95% confidence interval of [3.558,3.565]. This means there is a 95% chance the interval [3.558,3.565] contains the true parameter value of the slope. 
# These results indicate little uncertainty in the estimation of the slope of X. 
# Therefore, the business can be confident in the impact TV has on Sales.

# Potential areas to explore include:

    # Providing the business with the estimated sales given different TV promotional budgets
    # Using both TV and Radio as independent variables
    # Adding plots to help convey the results, such as using the seaborn regplot() to plot the data with a best fit regression line

# Of the three available promotion types (TV, radio, and social media), 
# TV has the strongest positive linear relationship with sales. 
# According to the model, an increase of one million dollars for the TV promotional budget will result in an estimated 3.5614 million dollars more in sales. 
# This is a very confident estimate, as the p-value for this coefficient estimate is small. 
# Thus, the business should prioritize increasing the TV promotional budget over the radio and social media promotional budgets to increase sales.
