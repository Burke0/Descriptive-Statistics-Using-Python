import pandas as pd
import seaborn as sns
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd # Import Tukey's HSD function


###############################
###### One-way ANOVA ##########
###############################

# Compares the means of one continuous dependent variable based on three or more groups of one categorical variable.


# Load in diamonds data set from seaborn package
diamonds = sns.load_dataset("diamonds")

# Examine first 5 rows of data set
print(diamonds.head())

# Check how many diamonds are each color grade
print(diamonds["color"].value_counts())

# Subset for colorless diamonds
colorless = diamonds[diamonds["color"].isin(["E","F","H","D","I"])]

# Select only color and price columns, and reset index
colorless = colorless[["color","price"]].reset_index(drop=True)

# Remove dropped categories of diamond color
colorless.color = colorless.color.cat.remove_categories(["G","J"])

# Check that the dropped categories have been removed
print(colorless["color"].values)

# Take the logarithm of the price, and insert it as the third column
colorless.insert(2, "log_price", [math.log(price) for price in colorless["price"]])

# Drop rows with missing values
colorless.dropna(inplace=True)

# Reset index
colorless.reset_index(inplace=True, drop=True)

# Examine first 5 rows of cleaned data set
print(colorless.head())

# Save to diamonds.csv
colorless.to_csv('diamonds.csv',index=False,header=list(colorless.columns))

# Save diamonds.csv as a variable called diamonds
diamonds = pd.read_csv("diamonds.csv")

# Examine first 5 rows of diamonds data set
print(diamonds.head())

# Create boxplot to show distribution of price by color grade
sns.boxplot(x = "color", y = "log_price", data = diamonds)

# Construct simple linear regression model, and fit the model
model = ols(formula = "log_price ~ C(color)", data = diamonds).fit()

# Get summary statistics
print(model.summary())

# Based on the model summary table, the color grades' associated beta coefficients all have a p-value of less than 0.05 (check the P>|t| column). But we can't be sure if there is a significant price difference between the various color grades. This is where one-way ANOVA comes in.

# Null Hypothesis
# 𝐻0:𝑝𝑟𝑖𝑐𝑒𝐷=𝑝𝑟𝑖𝑐𝑒𝐸=𝑝𝑟𝑖𝑐𝑒𝐹=𝑝𝑟𝑖𝑐𝑒𝐻=𝑝𝑟𝑖𝑐𝑒𝐼
# There is no difference in the price of diamonds based on color grade.

# Alternative Hypothesis
# 𝐻1:Not 𝑝𝑟𝑖𝑐𝑒𝐷=𝑝𝑟𝑖𝑐𝑒𝐸=𝑝𝑟𝑖𝑐𝑒𝐹=𝑝𝑟𝑖𝑐𝑒𝐻=𝑝𝑟𝑖𝑐𝑒𝐼
# There is a difference in the price of diamonds based on color grade.

# Run one-way ANOVA
sm.stats.anova_lm(model, typ = 2)
sm.stats.anova_lm(model, typ = 1)
sm.stats.anova_lm(model, typ = 3)

# Since all of the p-values (column PR(>F)) are very small, we can reject all three null hypotheses.


############################
###### Two-way ANOVA: ######
############################

# Compares the means of one continuous dependent variable based on three or more groups of two categorical variables.

# Import diamonds data set from seaborn package
diamonds = sns.load_dataset("diamonds")

# Examine first 5 rows of data set
print(diamonds.head())

# Subset for color, cut, price columns
diamonds2 = diamonds[["color","cut","price"]]

# Only include colorless diamonds
diamonds2 = diamonds2[diamonds2["color"].isin(["E","F","H","D","I"])]

# Drop removed colors, G and J
diamonds2.color = diamonds2.color.cat.remove_categories(["G","J"])

# Only include ideal, premium, and very good diamonds
diamonds2 = diamonds2[diamonds2["cut"].isin(["Ideal","Premium","Very Good"])]

# Drop removed cuts
diamonds2.cut = diamonds2.cut.cat.remove_categories(["Good","Fair"])

# Drop NaNs
diamonds2.dropna(inplace = True)

# Reset index
diamonds2.reset_index(inplace = True, drop = True)

# Add column for logarithm of price
diamonds2.insert(3,"log_price",[math.log(price) for price in diamonds2["price"]])

# Examine the data set
print(diamonds2.head())

# Save as diamonds2.csv
diamonds2.to_csv('diamonds2.csv',index=False,header=list(diamonds2.columns))

# Load the data set
diamonds2 = pd.read_csv("diamonds2.csv")

# Examine the first 5 rows of the data set
print(diamonds2.head())

# Construct a multiple linear regression with an interaction term between color and cut
model2 = ols(formula = "log_price ~ C(color) + C(cut) + C(color):C(cut)", data = diamonds2).fit()

# Get summary statistics
print(model2.summary())

# Based on the model summary table, many of the color grades' and cuts' associated beta coefficients have a p-value of less than 0.05 (check the P>|t| column). 
# Additionally, some of the interactions also seem statistically signifcant. 
# We'll use a two-way ANOVA to examine further the relationships between price and the two categories of color grade and cut.

# First, we have to state our three pairs of null and alternative hypotheses:

# 𝐻0:𝑝𝑟𝑖𝑐𝑒𝐷=𝑝𝑟𝑖𝑐𝑒𝐸=𝑝𝑟𝑖𝑐𝑒𝐹=𝑝𝑟𝑖𝑐𝑒𝐻=𝑝𝑟𝑖𝑐𝑒𝐼
# There is no difference in the price of diamonds based on color.
# 𝐻1:Not 𝑝𝑟𝑖𝑐𝑒𝐷=𝑝𝑟𝑖𝑐𝑒𝐸=𝑝𝑟𝑖𝑐𝑒𝐹=𝑝𝑟𝑖𝑐𝑒𝐻=𝑝𝑟𝑖𝑐𝑒𝐼
# There is a difference in the price of diamonds based on color.

# 𝐻0:𝑝𝑟𝑖𝑐𝑒𝐼𝑑𝑒𝑎𝑙=𝑝𝑟𝑖𝑐𝑒𝑃𝑟𝑒𝑚𝑖𝑢𝑚=𝑝𝑟𝑖𝑐𝑒𝑉𝑒𝑟𝑦 𝐺𝑜𝑜𝑑
# There is no difference in the price of diamonds based on cut.
# 𝐻1:Not 𝑝𝑟𝑖𝑐𝑒𝐼𝑑𝑒𝑎𝑙=𝑝𝑟𝑖𝑐𝑒𝑃𝑟𝑒𝑚𝑖𝑢𝑚=𝑝𝑟𝑖𝑐𝑒𝑉𝑒𝑟𝑦 𝐺𝑜𝑜𝑑
# There is a difference in the price of diamonds based on cut.

# 𝐻0:The effect of color on diamond price is independent of the cut, and vice versa.
# 𝐻1:There is an interaction effect between color and cut on diamond price.

# Run two-way ANOVA
print('two way anova: ',sm.stats.anova_lm(model2, typ = 2))
sm.stats.anova_lm(model2, typ = 1)
sm.stats.anova_lm(model2, typ = 3)

# Since all of the p-values (column PR(>F)) are very small, we can reject all three null hypotheses.

##############################
#### ANOVA post hoc test #####
##############################

# Post hoc test: Performs a pairwise comparison between all available groups while controlling for the error rate.

# Load in the data set from one-way ANOVA
diamonds = pd.read_csv("diamonds.csv")

# Here we follow the same steps as above:

#    Build a simple linear regression model
#    Check the results
#    Run one-way ANOVA

# Construct simple linear regression model, and fit the model
model = ols(formula = "log_price ~ C(color)", data = diamonds).fit()

# Get summary statistics
print(model.summary())

# Run one-way ANOVA
sm.stats.anova_lm(model, typ=2)

# Run Tukey's HSD post hoc test for one-way ANOVA
tukey_oneway = pairwise_tukeyhsd(endog = diamonds["log_price"], groups = diamonds["color"], alpha = 0.05)

# Get results (pairwise comparisons)
print(tukey_oneway.summary())

# Each row represents a pariwise comparison between the prices of two diamond color grades. 
# The reject column tells us which null hypotheses we can reject. 
# Based on the values in that column, we can reject each null hypothesis, except when comparing D and E color diamonds. 
# We cannot reject the null hypothesis that the diamond price of D and E color diamonds are the same.