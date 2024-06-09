# Diamond Price Prediction

This project aims to predict the price of diamonds based on various features such as carat, cut, color, clarity, and dimensions. The application is built using Python and several machine learning libraries. It employs an XGBoost model and is deployed using a comprehensive CI/CD pipeline involving Docker, Jenkins, Ansible, Minikube, and Streamlit.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [CI/CD Pipeline](#cicd-pipeline)
- [Model Evaluation](#model-evaluation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Diamond Price Prediction application allows users to predict the price of a diamond based on specific features. Users input these features through a Streamlit web interface, and the application outputs the predicted price.

## Features

- Predict diamond prices based on carat, cut, color, clarity, depth, table, length, width, and depth.
- User-friendly interface built with Streamlit.
- Robust CI/CD pipeline for automated testing, building, and deployment.
- Real-time predictions using a pre-trained RandomForestRegressor model.

## Tech Stack

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Joblib
- **CI/CD Tools:** Jenkins, Ansible
- **Containerization:** Docker
- **Orchestration:** Minikube (Kubernetes)
- **Deployment:** Minikube

## Installation

### Prerequisites

- Python 3.8+
- Docker
- Minikube
- Jenkins
- Ansible

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/diamond-price-prediction.git
   cd diamond-price-prediction
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Docker:**

   ```bash
   docker build -t diamond-price-prediction:latest .
   ```

4. **Deploy with Minikube:**

   ```bash
   kubectl apply -f Deployment.yml
   kubectl apply -f Service.yml
   minikube service diamond-prediction-service
   ```

## Usage

1. **Start the Application:**

   Run the following command to start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. **Enter Diamond Features:**

   Use the sidebar to input the features of the diamond.

3. **Get Price Prediction:**

   Click the "Predict" button to get the predicted price displayed on the main screen.

## CI/CD Pipeline

### Overview

The CI/CD pipeline for this project automates the process of testing, building, and deploying the diamond price prediction application. The pipeline is managed using Jenkins and involves multiple stages executed across three different servers.

### Stages

1. **Git Checkout:**

   - **Description:** The pipeline starts by checking out the project repository from GitHub to ensure the latest code is used.

2. **Show Downloaded Files:**

   - **Description:** The pipeline lists the files downloaded from the GitHub repository to verify that the correct files have been retrieved.

3. **SSH to Ansible Server and Copy Files:**

   - **Description:** The pipeline connects to the Ansible server and copies the necessary files from the Jenkins server to the Ansible server. This is to prepare for building the Docker image.

4. **Docker Image Building:**

   - **Description:** On the Ansible server, the pipeline builds a Docker image using the copied files. This image contains the application and its dependencies.

5. **Docker Image Tagging:**

   - **Description:** The built Docker image is tagged with a specific version and a latest tag to ensure version control and easy access.

6. **Push Docker Images to DockerHub:**

   - **Description:** The tagged Docker images are pushed to DockerHub, a container registry, to make them accessible for deployment.

7. **Copying Files to Kubernetes Server:**

   - **Description:** The pipeline then copies the necessary deployment files from the Jenkins server to the Kubernetes server, preparing for deployment in the Minikube cluster.

8. **Kubernetes Deployment Using Ansible:**

   - **Description:** Finally, the pipeline uses Ansible to deploy the application to the Kubernetes cluster, ensuring the application is up and running in the Minikube environment.

## Model Evaluation

The RandomForestRegressor model is evaluated using the following metrics:

- **R²:** Measures the proportion of variance in the dependent variable predictable from the independent variables.
- **Adjusted R²:** Adjusted version of R² that accounts for the number of predictors in the model.
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual values.
- **MSE (Mean Squared Error):** Average squared difference between predicted and actual values.
- **RMSE (Root Mean Squared Error):** Square root of the average squared difference between predicted and actual values.

## Contributing

Contributions are welcome! Please fork the repository and create a pull request.

## License

This project is licensed under the MIT License.