# Standard operational package imports.
import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics

# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns

###################################################################
############ Logistic Regression II: Electric Boogaloo ############
###################################################################

# Load the dataset 
df_original = pd.read_csv("Invistico_Airline.csv")

# Print first 10 rows
print('head:\n', df_original.head(10))

# Print data types
print('data types:\n', df_original.dtypes)

# print count of satisfied customers
print(df_original['satisfaction'].value_counts(dropna = False))

# Check for missing values
print('Null values:\n', df_original.isnull().sum())

# Create a subset of the dataframe without the missing values
df_subset = df_original.dropna(axis=0).reset_index(drop = True)

# sns.regplot needs float type instead of int
df_subset = df_subset.astype({"Inflight entertainment": float})

# Convert categorical data 'satisfaction' into numeric (one-hot encoding)
df_subset['satisfaction'] = OneHotEncoder(drop='first').fit_transform(df_subset[['satisfaction']]).toarray()

print(df_subset.head(10))

# Create train/test data
X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Build a logistic regression model and fit to the training data
clf = LogisticRegression().fit(X_train,y_train)

print('coefficient: ',clf.coef_,' intercept: ', clf.intercept_)

# Plot the model
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset, logistic=True, ci=None)
plt.show()
# The graph seems to indicate that the higher the inflight entertainment value, the higher the customer satisfaction, 
# though this is currently not the most informative plot. The graph currently doesn't provide much insight into the data points, as Inflight entertainment is categorical. 

# Save predictions.
y_pred = clf.predict(X_test)

# Use predict_proba to output a probability.
print(clf.predict_proba(X_test))

# Use predict to output 0's and 1's.
print(clf.predict(X_test))

# Print out the model's accuracy, precision, recall, and F1 score
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))

# Create a confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)
disp.plot()
plt.show() 
# Logistic regression accurately predicted satisfaction 80.2 percent of the time.
# The confusion matrix is useful, as it displays a similar amount of true positives and true negatives.

##### Key Takeaways ####
# Customers who rated in-flight entertainment highly were more likely to be satisfied. Improving in-flight entertainment should lead to better customer satisfaction.
# The model is 80.2 percent accurate. This is an improvement over the dataset's customer satisfaction rate of 54.7 percent.
# The success of the model suggests that the airline should invest more in model developement to examine if adding more independent variables leads to better results. 
# Building this model could not only be useful in predicting whether or not a customer would be satisfied but also lead to a better understanding of what independent variables lead to happier customers.
