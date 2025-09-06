🚀 Loan Default Prediction – Logistic Regression Capstone Project

📌 Project Overview

This project aims to predict loan default risk using Logistic Regression.
The model helps financial institutions assess whether a loan applicant is likely to repay or default before approving the loan.

The workflow covers the end-to-end ML lifecycle:

> Data preprocessing (handling missing values, encoding categorical variables, balancing with SMOTE)

> Model training & evaluation

> Saving model and scaler as pickle files

> Deployment via Flask API

> Containerization with Docker

🗂️ Dataset

Size: ~45,000+ records × 14 columns

Features:

person_age, person_gender, person_education, person_income, person_emp_exp

person_home_ownership, loan_amnt, loan_intent, loan_int_rate, loan_percent_income

cb_person_cred_hist_length, credit_score, previous_loan_defaults_on_file

Target:

loan_status → 1 = Default, 0 = No Default

⚠️ Dataset not included here (use data/loan_data.csv in your local setup).

🔧 Tech Stack

Language: Python 3.10

Libraries: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Flask, Requests, Pickle

Deployment: Flask API + Docker

⚙️ Project Workflow
1. Model Training (train.py)

Handles preprocessing:

Missing value imputation

Label Encoding for categorical features

Balancing dataset using SMOTE

Feature scaling with StandardScaler

Trains a Logistic Regression model

Saves:

loan_model.pkl → model + scaler

label_encoders.pkl → label encoders (for categorical mapping)

2. Flask API (app/app.py)

Endpoints:

/ → health check

/predict → accepts JSON input and returns prediction

Input format (example):

{
  "person_age": 22,
  "person_gender": "female",
  "person_education": "Master",
  "person_income": 71948,
  "person_emp_exp": 0,
  "person_home_ownership": "RENT",
  "loan_amnt": 35000,
  "loan_intent": "PERSONAL",
  "loan_int_rate": 16.02,
  "loan_percent_income": 0.49,
  "cb_person_cred_hist_length": 3,
  "credit_score": 561,
  "previous_loan_defaults_on_file": "No"
}


API response:

{ "prediction": 1 }

3. Docker Deployment

Dockerfile builds a container with:

Python 3.10-slim base image

Required dependencies (requirements.txt)

Trained model & API

Build the image:

docker build -t loan-prediction .


Run the container:

docker run -p 5000:5000 loan-prediction



📊 Results

Model: Logistic Regression

Metrics:

Accuracy: ~89%

Precision/Recall/F1: Balanced after applying SMOTE

Strengths:

Interpretable model

Good recall for defaults (catching risky loans)

Limitations:

Logistic Regression is linear → may miss non-linear patterns

## 📊 Architecture

<img width="2385" height="1516" alt="loan_prediction_architecture" src="https://github.com/user-attachments/assets/6a83fd1e-1052-4528-8277-d13d4d667e98" />



🔮 Future Improvements

Test advanced models (Random Forest, XGBoost, Neural Networks)

Build a dashboard for real-time loan risk visualization

Deploy on AWS/GCP/Azure for production scaling
