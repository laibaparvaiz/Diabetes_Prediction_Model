# Diabetes Prediction Model 🩺

An interactive, end-to-end machine learning web application that predicts the likelihood of diabetes based on medical predictor variables. 

This project utilizes a robust scikit-learn pipeline to handle data preprocessing (imputation and scaling) and a Logistic Regression classifier to deliver real-time predictions via a user-friendly Gradio interface. It is configured to run smoothly on local development environments or be deployed to cloud platforms like Railway.

## 🚀 Features
* Robust Preprocessing Pipeline: Automatically handles missing data (replacing `0` values with `NaN` and imputing the mean) to ensure higher model accuracy.
* Feature Scaling: Uses `StandardScaler` to normalize input features so the Logistic Regression model treats all data points equally.
* Interactive Web UI: Built with Gradio to provide a seamless, form-based interface for users to input medical data and see immediate results.
* Confidence Scoring: Outputs not just the prediction (High/Low Risk) but also the model's confidence percentage.
* CPU-Friendly & Lightweight: Highly optimized architecture that trains and runs efficiently on standard hardware without requiring a GPU.

## 🛠️ Tech Stack
* Language: Python
* Data Manipulation: Pandas, NumPy
* Machine Learning: Scikit-Learn (Logistic Regression, ColumnTransformer, Pipeline)
* **Web Framework:** Gradio

## 📊 Dataset
The model is trained on the classic "Pima Indians Diabetes Dataset". The target variable is `Outcome` (0 for non-diabetic, 1 for diabetic), evaluated against 8 medical predictor features such as BMI, Insulin levels, Age, and Glucose concentration.
