import pandas as pd

# Load and preprocess the dataset
# Load the diamonds dataset from a CSV file
data = pd.read_csv("./data_models/diamonds.csv")

# Drop the first column as it seems to be an index
data = data.drop(["Unnamed: 0"], axis=1)

# Remove rows with zero values in 'x', 'y', 'z' columns, as they are not valid
data = data.drop(data[data["x"] == 0].index)
data = data.drop(data[data["y"] == 0].index)
data = data.drop(data[data["z"] == 0].index)

# Remove outliers based on 'depth', 'table', 'x', 'y', 'z' columns
data = data[(data["depth"] < 75) & (data["depth"] > 45)]
data = data[(data["table"] < 80) & (data["table"] > 40)]
data = data[data["x"] < 30]
data = data[data["y"] < 30]
data = data[(data["z"] < 30) & (data["z"] > 2)]

# Save the cleaned data to a new CSV file
data.to_csv('./data_models/cleaned_data.csv', index=False)
