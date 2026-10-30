# Customer Voice Intelligence System

## Overview

The Customer Voice Intelligence System is a web-based application that analyses customer reviews and predicts whether a product is recommended or not recommended based on the review text.

It uses Natural Language Processing (NLP) techniques and a machine learning model to process customer feedback and generate predictions.

## Features

- Analyse customer reviews through a simple web interface.
- Preprocess text using NLP techniques such as tokenisation, stop-word removal and lemmatisation.
- Convert review text into numerical features using TF-IDF.
- Predict whether a review is recommended or not recommended.
- Display prediction confidence percentages.

## Technologies Used

- **Programming Language:** Python
- **Web Framework:** Flask
- **Machine Learning:** Scikit-learn
- **NLP:** NLTK
- **Feature Extraction:** TF-IDF
- **Frontend:** HTML, CSS
- **Machine Learning Model:** Support Vector Machine (SVM)

## Project Structure

```text
Customer Voice Intelligence System/
│
├── app.py
├── requirements.txt
├── svm_model.pkl
├── tfidf_vectorizer.pkl
│
├── static/
│   └── fashion.jpg
│
├── templates/
│   └── index.html
│
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the project directory

```bash
cd "Customer Voice Intelligence System"
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000/
```

## How It Works

1. The user enters a customer review.
2. The application preprocesses the review using NLP techniques.
3. The processed text is transformed into numerical features using the TF-IDF vectorizer.
4. The trained SVM model analyses the features and predicts the recommendation category.
5. The application displays the prediction along with its confidence percentage.

## Note

The trained SVM model and TF-IDF vectorizer are included in the repository as pickle files. These files are required for the application to make predictions.