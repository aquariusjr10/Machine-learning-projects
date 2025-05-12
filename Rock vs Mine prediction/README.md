# 🪨 Rock vs Mine Prediction using Machine Learning

This project focuses on building a machine learning model to classify sonar signal returns as either **rock** or **mine**. The signals are used to detect underwater objects like submarines or rocks using sonar reflection data.

---

## 📌 Project Overview

- **Objective**: Predict whether an object is a *rock* or a *mine* based on sonar signal data.
- **Approach**: Use logistic regression on numerical sonar readings.
- **Dataset Source**: [UCI Machine Learning Repository – Sonar Dataset](https://archive.ics.uci.edu/ml/datasets/connectionist+bench+sonar+mines+vs+rocks)

---

## 📁 Dataset Details

- File: `sonar_data.csv`
- 208 samples, 60 features (numerical values representing sonar signal amplitudes)
- Label (column 60):  
  - `M` → Mine  
  - `R` → Rock

---

## 🛠️ Tools & Libraries

- Python
- Pandas & NumPy
- scikit-learn (for modeling and evaluation)
- Matplotlib (optional, for plotting)

---

## 🧪 Project Workflow

1. **Data Loading & Exploration**
   - Read CSV file into a DataFrame
   - Checked shape and label distribution

2. **Preprocessing**
   - Split dataset into features and target labels
   - Encode categorical labels (`M` and `R`) into binary format

3. **Model Training**
   - Split into training and test sets (stratified)
   - Trained a **Logistic Regression** model

4. **Evaluation**
   - Evaluated model using **accuracy score**
   - Verified performance on both training and test datasets

5. **Model Saving**
   - Saved the trained model using `pickle` for future inference

---

## 📈 Results

- The model achieved strong classification performance.
- Balanced accuracy across both rock and mine categories.
- Logistic Regression proved effective for this binary classification task.