🌧️ Rain Tomorrow Prediction - Australia
📌 Project Overview
This project aims to predict whether it will rain tomorrow in locations across Australia, based on 10 years of historical weather data. Using classification models, we aim to answer the question: "Should you carry an umbrella tomorrow?"

The target variable is RainTomorrow, a binary column indicating whether there was 1mm or more rain the next day.

📁 Dataset
Sourced from Australian Bureau of Meteorology

~10 years of daily weather observations from various Australian cities

Provided on Kaggle

🔑 Key Features
Date: Daily observation date

Location: Name of the weather station

MinTemp / MaxTemp: Temperature in °C

Rainfall / Evaporation / Sunshine: Weather readings

WindGustDir / WindDir9am / WindDir3pm

Humidity, Pressure, Temperature at 9am and 3pm

RainToday / RainTomorrow: Whether it rained that day/tomorrow (Yes/No)

🔍 Workflow
1. 🧠 Data Understanding
Data types and structure

Summary statistics (numerical and categorical)

Class imbalance check

2. 📊 Exploratory Data Analysis (EDA)
Univariate: Distributions and counts

Bivariate: Correlation heatmaps, scatter plots

Multivariate: Pair plots and trend analysis

3. 🧹 Data Preprocessing
Handling missing values and duplicates

Encoding categorical variables

Outlier treatment

Scaling features

Dealing with class imbalance

4. 🧠 Model Building
Models used:

Logistic Regression

Decision Trees / Random Forest

Gradient Boosting (e.g., XGBoost)

Performance metrics:

Accuracy, Precision, Recall, F1-Score

Confusion Matrix

5. 📈 Evaluation & Interpretation
Feature importance visualization

ROC-AUC curve

Comparison of model performances

🛠️ Tech Stack
Python (Jupyter Notebook)

Pandas, NumPy

Matplotlib, Seaborn

Scikit-learn, XGBoost
