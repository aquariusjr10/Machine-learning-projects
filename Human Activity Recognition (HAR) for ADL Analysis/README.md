# Human Activity Recognition (HAR) using Machine Learning & Deep Learning

This repository contains a comprehensive project for **Human Activity Recognition (HAR)** using the **UCI HAR dataset**. The project demonstrates how to preprocess sensor data, extract features, train multiple machine learning models (Logistic Regression, KNN, SVM, Random Forest, XGBoost) and a deep learning model (LSTM), and compare their performance for recognizing human activities.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)

## Project Overview

Human Activity Recognition (HAR) aims to classify activities (e.g., walking, sitting, standing) based on time-series sensor data collected from smartphones or wearable devices. This project:
- **Preprocesses** the UCI HAR dataset.
- **Trains** multiple ML models:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting (XGBoost)
- **Develops** a deep learning model using LSTM.
- **Evaluates** and compares model performances.

## Dataset

The project uses the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which contains:
- Accelerometer and gyroscope readings.
- Pre-processed training and testing sets.
- 6 Activity classes (e.g., Walking, Sitting, Standing, etc.).

## Project Structure

```plaintext
human-activity-recognition/
├── dataset/                                                               # UCI HAR dataset files (train/test folders)
├── models/                                                                # Directory to save trained models
├── Human Activity Recognition (HAR) for ADL Analysis.ipynb                # Jupyter Notebooks for Exploratory Data Analysis (EDA)
└── README.md                                                              # Project documentation (this file)