import pandas as pd
from sklearn.model_selection import train_test_split

# Load the cleaned dataset from a CSV file
label_data = pd.read_csv('./data_models/cleaned_data.csv')

# Split the data into features (X) and target variable (y)
X = label_data.drop(["price"], axis=1)
y = label_data["price"]

# Split the data into training and testing sets
# 75% of the data will be used for training and 25% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

# Save the training and testing sets to new CSV files
X_train.to_csv('./data_models/X_train.csv', index=False)
X_test.to_csv('./data_models/X_test.csv', index=False)
y_train.to_csv('./data_models/y_train.csv', index=False)
y_test.to_csv('./data_models/y_test.csv', index=False)
