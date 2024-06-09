import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the training data from CSV files
X_train = pd.read_csv('./data_models/X_train.csv')
y_train = pd.read_csv('./data_models/y_train.csv').values.ravel()

# Define the numeric and categorical features
numeric_features = ['carat', 'depth', 'table', 'x', 'y', 'z']
categorical_features = ['cut', 'color', 'clarity']

# Create a pipeline for numeric features with scaling
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

# Create a pipeline for categorical features with one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine the numeric and categorical transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a pipeline with the preprocessor and a RandomForestRegressor
pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())])

# Train the pipeline with the training data
pipeline_rf.fit(X_train, y_train)

# Save the trained pipeline to a file
joblib.dump(pipeline_rf, './data_models/random_forest_pipeline.pkl')
