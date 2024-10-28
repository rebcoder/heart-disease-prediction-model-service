# Heart Disease Prediction Model Service

This repository contains a Python-based machine learning model service that predicts heart disease based on health metrics. The model is served as an API using Flask, allowing integration with other services, such as a frontend or backend that collects user inputs.

## Features

- Uses a machine learning model (e.g., XGBoost) to predict the likelihood of heart disease based on 13 health metrics.
- Flask API to handle incoming prediction requests and return a prediction response.
- Model pipeline includes data preprocessing and one-hot encoding for categorical variables.

## Prerequisites

- Python 3.7 or higher
- Pipenv (for virtual environment and dependency management) or standard `pip`
- `model.pkl` (trained model file) - Ensure this file is placed in the project directory or specify the path.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rebcoder/heart-disease-prediction-model-service.git
cd heart-disease-prediction-model-service
2. Run the Flask API
python prediction_service.py

The service will start on http://localhost:5000 by default.
