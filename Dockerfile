# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /home/ubuntu/

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the CSV file into the container
COPY diamonds.csv /usr/src/app/diamonds.csv

# Run the preprocessing script
RUN python preprocessing.py

# Run the train-test splitting script
RUN python train_test_splitting.py

# Run the training script
RUN python training.py

# Run the testing script
CMD ["streamlit", "run", "testing.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Exposing the port
EXPOSE 80
