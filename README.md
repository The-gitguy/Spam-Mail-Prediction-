# Spam Mail Prediction

A small machine learning project to classify messages (email/SMS) as spam or not spam (ham). This repository provides a reproducible pipeline for data preprocessing, feature extraction, model training, evaluation, and inference using classic text classification techniques.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Train a model](#train-a-model)
  - [Evaluate a model](#evaluate-a-model)
  - [Run inference](#run-inference)
- [Modeling Notes](#modeling-notes)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Spam Mail Prediction is a straightforward supervised learning project to classify messages as `spam` or `ham`. It demonstrates a minimal, reproducible pipeline: load data, clean and vectorize text, train classifiers, evaluate, and serialize models for inference.

Typical approaches used in the project:
- Text cleaning and normalization
- TF-IDF vectorization with n-grams
- Classical classifiers: Multinomial Naive Bayes, Logistic Regression, Random Forest
- Standard evaluation metrics: accuracy, precision, recall, F1-score, ROC-AUC

## Features

- End-to-end example for training and predicting spam
- Reproducible scripts and/or notebook examples
- Model serialization for reuse
- Simple evaluation and reporting

## Dataset

Expect a CSV/TSV file with at least two columns:
- `label` — the class label (e.g. `spam` / `ham` or `1` / `0`)
- `text` — raw message contents (subject + body or full message)

Public datasets to try:
- UCI SMS Spam Collection
- Enron spam dataset
- Any labeled email/SMS dataset with `text` + `label`

## Requirements

- Python 3.8+
- Typical packages (install from requirements.txt if provided):
  - numpy, pandas, scikit-learn, joblib
  - nltk or spaCy (optional, for advanced preprocessing)

Install example:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Adjust file paths and parameters to your environment.

Train a model (example script):
```bash
python scripts/train.py --data data/spam.csv --model-out models/spam_clf.joblib
```

Evaluate a model:
```bash
python scripts/evaluate.py --data data/test.csv --model models/spam_clf.joblib
```

Predict / run inference:
```bash
python scripts/predict.py --model models/spam_clf.joblib --input "Win a free iPhone by clicking here"
```

Example expected evaluation output:
```
Accuracy: 0.98
Precision: 0.96
Recall: 0.95
F1-score: 0.955
ROC-AUC: 0.99
```

## Modeling Notes

- Clean text: lowercase, strip HTML, remove punctuation, normalize whitespace.
- Consider removing or customizing stopwords (sometimes keeping them helps for short messages).
- TF-IDF with uni-grams and bi-grams (ngram_range=(1,2)) is a good baseline.
- MultinomialNB is fast and strong for bag-of-words; Logistic Regression with class-weight or calibration often gives better probabilities.
- Use stratified splits and cross-validation when tuning hyperparameters.

## Project Structure (suggested)
````markdown
```text
Spam-Mail-Prediction-/
├── data/                  # raw and processed datasets
├── models/                # serialized models (.joblib / .pkl)
├── notebooks/             # EDA and experiments
├── scripts/               # train.py, evaluate.py, predict.py
├── src/                   # package: preprocessing, training, utils
├── requirements.txt
└── README.md
```
