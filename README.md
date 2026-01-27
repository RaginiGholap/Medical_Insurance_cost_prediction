# 🧠Medical Insurance Cost Estimator
AI-Driven Insurance Charges Prediction using Machine Learning

This project is a Machine Learning web application that predicts medical insurance costs based on a person's health and demographic details. It helps estimate insurance premiums using real-world data and regression models.

## 📌 Project Overview
Insurance companies calculate premiums based on many risk factors.
This project uses Machine Learning to automate and improve that process.

The model predicts insurance charges using features like:

Age

BMI

Smoking status

Number of children

Region

Gender

The final model is deployed as a Streamlit web app where users can input details and get an estimated insurance cost instantly.

## 🎯 Problem Statement
Manual insurance cost calculation:
Takes time
Is prone to human error
Cannot easily handle large data
This project builds a data-driven ML model that predicts medical insurance charges more accurately and efficiently.

## 📂 Dataset Information
Dataset Name: insurance.csv
Total Records: 1338
Total Features: 6 input features + 1 target
Target Variable: charges (medical insurance cost)

## Features Used
Feature	Description
age	Age of the person
sex	Gender (male/female)
bmi	Body Mass Index
children	Number of dependents
smoker	Smoking status (yes/no)
region	Residential region
charges	Medical insurance cost (Target)

## 📊 Exploratory Data Analysis (EDA)
Key insights from the data:
Smokers have significantly higher insurance charges
Higher BMI leads to higher medical costs
Older individuals tend to have higher charges
Charges distribution is right-skewed

## 🧹 Data Preprocessing
The following steps were performed before training the model:
Checked for missing values → No missing data
Converted categorical features into numeric:
Sex: male = 1, female = 0
Smoker: yes = 1, no = 0
Region encoded using one-hot encoding
Data split into:
80% Training data
20% Testing data
Feature scaling applied using StandardScaler

## 🤖 Model Building
Multiple regression models were tested:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Support Vector Regressor (SVR)

## 📈 Model Evaluation Metrics
R² Score – Measures how well predictions match actual values
MAE (Mean Absolute Error) – Average prediction error
RMSE (Root Mean Squared Error) – Penalizes large errors

## 🏆 Best Model
Random Forest Regressor performed best with the highest R² score and stable predictions.

## 🚀 Deployment
The model is deployed using Streamlit.

## 👤 User Inputs in Web App
-Age
BMI
Number of Children
Sex
Smoker Status
Region

## 💡 Output
Predicted Medical Insurance Cost

## 🛠 Tech Stack
Python
Pandas & NumPy – Data processing
Matplotlib & Seaborn – Data visualization
Scikit-learn – Machine learning models
Streamlit – Web app deployment
Pickle – Model saving

## 📷 Project Workflow
Data Collection → Data Cleaning → EDA → Preprocessing → Model Training → Model Evaluation → Deployment

## 🌍 Real-World Applications
Insurance premium estimation
Risk assessment
Healthcare cost analysis
Financial planning tools

## 🔮 Future Improvements
Add more health-related features (medical history, lifestyle)
Use advanced models like XGBoost or Neural Networks
Deploy on cloud platforms (AWS / Azure / GCP)
Improve UI/UX of the web application

## 🙌 Conclusion
This project demonstrates how Machine Learning can be used in the insurance industry to make fair, data-driven decisions and automate cost prediction.
