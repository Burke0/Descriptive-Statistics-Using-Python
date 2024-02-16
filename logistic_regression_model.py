import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load in sci-kit learn functionis for constructing logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import sklearn.metrics as metrics

# Load csv
activity = pd.read_csv("activity.csv")

# Get summary statistics about the dataset
print(activity.describe())

# Examine the dataset
print(activity.head())

#########################################################
##### Construct binomial logistic regression model ######
#########################################################

# Save X and y data into variables for train_test_split function
X = activity[["Acc (vertical)"]]
y = activity[["LyingDown"]]

# Split dataset into training and holdout datasets
# test_size 0.3 means the holdout dataset is only 30% of the total data that we have
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Build our classifier and fit the model to the data
clf = LogisticRegression().fit(X_train,y_train)

# Print the coefficient
print('The coefficient is: ', clf.coef_)

# Print the intercept
print('The intercept is: ', clf.intercept_)

# Our model has an intercept or 𝛽0 of 6.10 and a 𝛽1 of -0.12. 

# Plot the logistic regression and its confidence band  
# logistic=true so the function knows were plotting a logistic regression model, not a linear regression one
# This finds the best way to determine the likelihood of someone lying down based on vertical acceleration
sns.regplot(x="Acc (vertical)", y="LyingDown", data=activity, logistic=True)
plt.show()
plt.close()

#################################
####### Confusion matrix ########
#################################

# Save predictions
y_pred = clf.predict(X_test)

# Print out the predicted labels, 0 means not lying down, and 1 means lying down. 
print('Predicted labels: ', clf.predict(X_test))

# Print out the predicted probabilities
print('Predicted probabilities: ', clf.predict_proba(X_test)[::,-1])

# Calculate the values for each quadrant in the confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)

# Create the confusion matrix as a visualization
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)

# Display the confusion matrix
# The upper-left quadrant displays the number of true negatives. The number of people that were not lying down that the model accurately predicted were not lying down.
# The bottom-left quadrant displays the number of false negatives. The number of people that were lying down that the model inaccurately predicted were not lying down.
# The upper-right quadrant displays the number of false positives. The number of people that were not lying down that the model inaccurately predicted were lying down.
# The bottom-right quadrant displays the number of true positives. The number of people that were lying down that the model accurately predicted were lying down.
# A perfect model would yield all true negatives and true positives, and no false negatives or false positives.
disp.plot()
plt.show()
