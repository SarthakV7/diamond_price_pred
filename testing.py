import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics
import joblib

# Load the training and testing data
X_train = pd.read_csv('./data_models/X_train.csv')
X_test = pd.read_csv('./data_models/X_test.csv')
y_train = pd.read_csv('./data_models/y_train.csv').values.ravel()
y_test = pd.read_csv('./data_models/y_test.csv').values.ravel()

# Load pre-trained model
pipeline_rf = joblib.load('./data_models/random_forest_pipeline.pkl')

# Make predictions on the test data
pred = pipeline_rf.predict(X_test)
print(X_test.shape)

# Model Evaluation
print("R^2:", metrics.r2_score(y_test, pred))
print("Adjusted R^2:", 1 - (1 - metrics.r2_score(y_test, pred)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1))
print("MAE:", metrics.mean_absolute_error(y_test, pred))
print("MSE:", metrics.mean_squared_error(y_test, pred))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, pred)))

# Function to predict the price based on input features
def predict_price(features):
    df = pd.DataFrame([features], columns=[
        'carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    prediction = pipeline_rf.predict(df)
    return prediction[0]

# Streamlit App
st.title("Diamond Price Prediction")

# Sidebar for input features
st.sidebar.header("Enter the diamond features below:")

# Organize inputs for categorical variables
st.sidebar.subheader("Categorical Variables")
cut = st.sidebar.selectbox("Cut", ["Ideal", "Premium", "Good", "Very Good", "Fair"])
color = st.sidebar.selectbox("Color", ["E", "I", "J", "H", "F", "G", "D"])
clarity = st.sidebar.selectbox("Clarity", ["SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"])

# Organize inputs for numerical variables
st.sidebar.subheader("Numerical Variables")
carat = st.sidebar.number_input("Carat", min_value=0.0, value=0.23)
depth = st.sidebar.number_input("Depth", min_value=0.0, value=61.5)
table = st.sidebar.number_input("Table", min_value=0.0, value=55.0)
length = st.sidebar.number_input("Length (in mm)", min_value=0.0, value=3.95)
width = st.sidebar.number_input("Width (in mm)", min_value=0.0, value=3.98)
depth_in_mm = st.sidebar.number_input("Depth (in mm)", min_value=0.0, value=2.43)

# Predict the price when the button is clicked
if st.sidebar.button("Predict"):
    features = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': length,
        'y': width,
        'z': depth_in_mm
    }

    # Convert the features to a numpy array for prediction
    features_values = np.array(list(features.values())).reshape(1, -1)[0]
    prediction = predict_price(features_values)
    
    st.subheader(f"The predicted price of the diamond is: ${prediction}")
