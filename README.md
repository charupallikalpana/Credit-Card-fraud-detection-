# Credit Card Fraud Detection

A machine learning project that detects fraudulent credit card transactions using **Logistic Regression** and **XGBoost**, with **SMOTE** to handle the heavy class imbalance.

## Overview

Fraud is rare compared to normal transactions, so plain accuracy is misleading (a model that always predicts "Non-Fraud" would still score very high). This project focuses on **precision, recall, F1-score and the confusion matrix**, and uses:

- Exploratory data analysis (class distribution, correlations, time and amount distributions, outliers)
- Duplicate removal and missing-value handling
- Stratified train/test split (80/20) to keep the fraud ratio consistent
- SMOTE oversampling on the training set only
- Two models: Logistic Regression and XGBoost
- Evaluation with accuracy, precision, recall, classification report, confusion matrix and 5-fold stratified cross-validation

## Dataset

File: `creditcard_1_1.csv` (19,219 rows, 31 columns)

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since the first transaction in the dataset |
| `V1` - `V28` | Anonymized numerical features (PCA-transformed) |
| `Amount` | Transaction amount |
| `Class` | Target label: `1` = fraud, `0` = non-fraud |

> Add the source of the dataset here (for example, the Kaggle page you downloaded it from).

## Project Structure

```
.
├── credit_card_fraud_detection.py   # Main script (EDA, training, evaluation)
├── creditcard_1_1.csv               # Dataset
├── requirements.txt                 # Python dependencies
├── .gitignore
└── README.md
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
   ```

2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Make sure `creditcard_1_1.csv` is in the same folder as the script, then run:

```bash
python credit_card_fraud_detection.py
```

The script will print the data summary and model metrics, and open the plots (class distribution, correlation heatmap, time distributions, boxplots) one by one.

The script was exported from a Google Colab notebook. If the CSV is not found locally, it falls back to the Colab upload dialog, which only works inside Colab. When running locally, keep the CSV next to the script.

## Models

| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline model trained on SMOTE-resampled data |
| XGBoost | `n_estimators=500`, `learning_rate=0.1`, `max_depth=4`, with subsampling and L1/L2 regularization to limit overfitting |

## Results

Add your final metrics here (precision, recall, F1-score for the fraud class, and the confusion matrix) after running the script.

## Tech Stack

Python, pandas, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn, plotly

## License

Add a license here (for example MIT), or remove this section if the repository is private.
