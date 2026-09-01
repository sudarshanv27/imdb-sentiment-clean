# 🎬 IMDb Sentiment Analysis

An end-to-end Machine Learning project that analyzes IMDb movie reviews and predicts whether a review expresses **Positive** or **Negative** sentiment.

The project includes data preprocessing, TF-IDF feature extraction, multiple machine learning models, evaluation, a Streamlit web application, a FastAPI REST API, automated tests, and Render deployment.

---

## 🚀 Live Demo

### 🌐 Streamlit Web Application

👉 **Live App:**  
https://imdb-sentiment-app-eltv.onrender.com/

Use the web application to enter any movie review and receive a sentiment prediction with confidence and probability scores.

### 🔌 FastAPI REST API

👉 **API:**  
https://imdb-sentiment-api-8eok.onrender.com

### 📚 Swagger API Documentation

👉 **API Docs:**  
https://imdb-sentiment-api-8eok.onrender.com/docs


---

# 📌 Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone of text.

In this project, IMDb movie reviews are classified into:

- 🟢 **Positive**
- 🔴 **Negative**

The final system uses:

**TF-IDF → Logistic Regression → Sentiment Prediction**

The trained model achieved:

> **89.43% accuracy on the final IMDb test dataset.**

---

# ✨ Features

- 📊 IMDb movie review sentiment classification
- 🧹 Text preprocessing
- 🔤 TF-IDF feature extraction
- 🤖 Multiple machine learning models
- 📈 Model performance comparison
- 🎯 Final model selection
- 📊 Confusion matrix
- 📋 Classification report
- 🌐 Streamlit web interface
- 🔌 FastAPI REST API
- 📚 Swagger API documentation
- 🧪 Automated unit and API testing
- 🚀 Render deployment
- 💾 Saved trained models using Joblib

---

# 🛠️ Tech Stack

## Programming Language

- Python 3.13

## Machine Learning

- Scikit-learn
- Logistic Regression
- Linear SVM
- Multinomial Naive Bayes

## NLP

- TF-IDF Vectorization
- Text preprocessing

## Data Processing

- Pandas
- NumPy

## Visualization

- Matplotlib

## Web Application

- Streamlit

## REST API

- FastAPI
- Uvicorn
- Pydantic

## Testing

- Pytest
- HTTPX

## Model Serialization

- Joblib

## Deployment

- GitHub
- Render

---

# 📂 Project Structure

```text
imdb-sentiment-clean/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   └── external/
│
├── models/
│   ├── logistic_regression.pkl
│   ├── linear_svm.pkl
│   ├── naive_bayes.pkl
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/
│   └── EDA.ipynb
│
├── reports/
│   ├── accuracy_report.txt
│   ├── classification_report.txt
│   └── confusion_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── config.py
│   ├── dashboard.py
│   ├── evaluate.py
│   ├── feature_extraction.py
│   ├── predict.py
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_predict.py
│
├── app.py
├── main.py
├── render.yaml
├── requirements.txt
├── .gitignore
└── README.md