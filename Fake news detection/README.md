# 📰 Fake News Detection with Machine Learning

This project uses natural language processing (NLP) and machine learning to detect whether a news article is **real** or **fake** based on its textual content.

---

## 📌 Project Overview

- **Goal**: Build a binary classifier that predicts if a news article is fake (1) or real (0).
- **Approach**: Text preprocessing → TF-IDF vectorization → Logistic Regression model.
- **Dataset Source**: [Kaggle – Fake News Classification](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)

---

## 📁 Dataset Information

- File: `Fake_Dataset.csv`
- Key features:
  - `title`: Headline of the article
  - `text`: Main content
  - `label`: Target variable (1 = fake, 0 = real)

---

## 🛠️ Tools & Libraries

- Python
- Pandas & NumPy
- NLTK (for stopword removal and stemming)
- Scikit-learn (for model building and evaluation)
- Jupyter Notebook

---

## 🧪 Project Steps

1. **Data Loading**  
   Load CSV file and inspect its structure.

2. **Text Preprocessing**  
   - Remove punctuation and non-alphabetic characters  
   - Convert to lowercase  
   - Remove stopwords using NLTK  
   - Apply Porter Stemmer

3. **Feature Extraction**  
   - Convert cleaned text into numerical vectors using **TF-IDF**

4. **Model Training**  
   - Use **Logistic Regression** as the classification model  
   - Train-test split with scikit-learn

5. **Evaluation**  
   - Accuracy score  
   - Model performance on unseen test data

---

## 📈 Results

- Logistic Regression performed well with high accuracy.
- TF-IDF captured important text features for classification.
- The model distinguishes fake from real news based solely on article content.

---

## 🚀 Getting Started

### Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
