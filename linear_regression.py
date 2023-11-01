import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from scipy.stats import shapiro

# Load dataset
penguins = sns.load_dataset("penguins")

# Examine first 5 rows of dataset
print(penguins.head())

# From the first 5 rows of the dataset, we can see that there are several columns available: 
# species, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, and sex. 
# There also appears to be some missing data.

# Keep Adelie and Gentoo penguins, drop missing values
penguins_sub = penguins[penguins["species"] != "Chinstrap"]

# dropna() function by default removes any rows with any missing values in any of the columns
penguins_final = penguins_sub.dropna()

# reset_index() function resets the index values for the rows in the dataframe
# inplace=True, you will not create a new DataFrame object
# drop=True, you will not insert a new index column into the DataFrame object.
penguins_final.reset_index(inplace=True, drop=True)

# Create pairwise scatterplots of data set
sns.pairplot(penguins_final)
plt.show()

# From the scatterplot matrix, you can observe a few linear relationships:
# 1) bill length (mm) and flipper length (mm)
# 2) bill length (mm) and body mass (g)
# 3) flipper length (mm) and body mass (g)

# Subset Data to focus on linear relationship2(bill length (mm) and body mass (g))
ols_data = penguins_final[["bill_length_mm", "body_mass_g"]]

# Write out formula (ols = dependent variable(y) ~ independent variable(x))
ols_formula = "body_mass_g ~ bill_length_mm"

# Build OLS, fit model to data
OLS = ols(formula = ols_formula, data = ols_data)

# Fit the model to the data
model = OLS.fit()

# Get the coefficients and more statistics about the model to evaluate and interpret the results
print(model.summary())

# You can use the regplot() function from seaborn to visualize the regression line.
sns.regplot(x = "bill_length_mm", y = "body_mass_g", data = ols_data)

#############################################
##### Finish checking model assumptions #####
#############################################

# 1) Linearity (The scatterplot matrix satisfied this assumption)
# 2) Independent observations (There is no reason to believe that one penguin's body mass or bill length would be related to any other penguin's anatomical measurements)
# 3) Normality
# 4) Homoscedasticity

# Normality - we must check residuals, as an approximation of the errors

# Subset X variable
X = ols_data[["bill_length_mm"]]

# Get predictions from model
fitted_values = model.predict(X)

# Calculate residuals
residuals = model.resid

# We can create a histogram of the residuals to visually check for normality
# Since the residuals are close to normal distribution, the assumtion is met.
plt.clf()
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals(Penguin dataset model)")
plt.show()

# another way is check a Q-Q plot for a straight diagonal line starting from the bottom left corner(shows normal distribution)
fig = sm.qqplot(model.resid, line = 's')
plt.show()

# To confirm we can perform the Shapiro-Wilk test for normality
shapiro_test_statistic, shapiro_p_value = shapiro(residuals)

# Display the results of the Shapiro-Wilk test
print("Shapiro-Wilk Test:")
print(f"Test Statistic: {shapiro_test_statistic}")
print(f"P-Value: {shapiro_p_value}")

# Check for normality based on the Shapiro-Wilk test
if shapiro_p_value > 0.05:
    print("Residuals appear to be normally distributed (p > 0.05).")
else:
    print("Residuals do not appear to be normally distributed (p ≤ 0.05).")


# Homoscedasticity - To check the homoscedasticity assumption, we can create a scatterplot of the fitted values and residuals. 
# If the plot resembles a random cloud (i.e., the residuals are scattered randomly), then the assumption is likely met.

fig = sns.scatterplot(x=fitted_values, y=residuals)

# Add reference line at residuals = 0
fig.axhline(0)

# Set x-axis and y-axis labels
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")

plt.show()