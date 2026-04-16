## End to End ML Project
Student Exam Performance Indicator – End to End ML Project
📌 Project Overview

The Student Exam Performance Indicator project aims to predict a student’s math score based on various features such as:

Gender
Race/Ethnicity
Parental level of education
Lunch type
Test preparation course
Reading score
Writing score

This is a complete end-to-end Machine Learning project including:

Data Ingestion
Data Preprocessing
Exploratory Data Analysis (EDA)
Model Training
Model Evaluation
Model Deployment
🚀 Problem Statement

Educational institutions want to analyze student performance and identify factors affecting exam results.

This project builds a regression model to predict the math score of a student using other academic and demographic features.

Project Architecture
Student-Exam-Performance/
│
├── artifacts/                  # Saved models & processed files
├── notebooks/                  # EDA & experiments
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │
│   ├── pipeline/
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/                  # HTML templates (for Flask app)
├── static/                     # CSS files
├── app.py                      # Flask application
├── requirements.txt
└── README.md

🔍 Exploratory Data Analysis (EDA)

Performed:

Distribution analysis
Correlation heatmap
Feature relationship analysis
Outlier detection

Key Observations:

Reading & Writing scores strongly correlate with Math score.
Test preparation course positively impacts performance.
Parental education level influences scores.
🤖 Machine Learning Models Used

The following regression models were trained and compared:

Linear Regression
Ridge Regression
Lasso Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor

Best model selected based on:

R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)

🛠️ Tech Stack
Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Flask
HTML/CSS
Git & GitHub
