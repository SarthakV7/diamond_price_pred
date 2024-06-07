import pandas as pd
from sklearn.model_selection import train_test_split

label_data = pd.read_csv('cleaned_data.csv')

X = label_data.drop(["price"], axis =1)
y = label_data["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7)

X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)