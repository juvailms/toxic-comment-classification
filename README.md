# Toxic Comment Classification (Multi-Label)

## Overview

This project implements a complete end-to-end multi-label text classification pipeline for the **Jigsaw Toxic Comment Classification Challenge**.

Given a user comment, the model predicts whether it belongs to one or more of the following toxicity categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The solution uses **TF-IDF Vectorization** for feature extraction and **One-vs-Rest Logistic Regression** for multi-label classification.

---

# Project Structure

```
folder/
│
├── preprocessing.py
├── train.py
├── evaluate.py
├── README.md
├── Model_Report.docx
├── Reproducibility_Checklist.docx
├── requirements.txt
│
└── outputs/
    ├── per_label_metrics.csv
    └── overall_metrics.csv
```

---

# Dataset

Dataset:

**Jigsaw Toxic Comment Classification Challenge**

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Place the downloaded **train.csv** file in the project root directory before running the code.

**Note:** The dataset is **not included** in this submission.

---

# Model

Feature Extraction

- TF-IDF Vectorizer
- Unigrams and Bigrams
- Maximum Features = 50,000
- Minimum Document Frequency = 2

Classifier

- Logistic Regression
- One-vs-Rest Strategy
- Balanced Class Weights
- Maximum Iterations = 1000

---

# Preprocessing

The preprocessing pipeline performs:

- Convert text to lowercase
- Remove URLs
- Remove HTML tags
- Replace newline and tab characters
- Remove extra whitespace
- TF-IDF vectorization

---

# Data Split

The dataset is divided into:

| Dataset | Percentage |
|----------|-----------:|
| Train | 70% |
| Validation | 10% |
| Test | 20% |

A fixed random seed (`random_state = 42`) is used to ensure reproducibility.

---

# Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# How to Run

## Step 1

Download the dataset from Kaggle.

Place:

```
train.csv
```

inside the project folder.

---

## Step 2

Train the model:

```bash
python train.py
```

This will:

- Load and preprocess the dataset
- Train the One-vs-Rest Logistic Regression model
- Generate predictions on the test set
- Save prediction results to the `outputs` folder

---

## Step 3

Evaluate the model:

```bash
python evaluate.py
```

This will:

- Evaluate the trained model
- Compute per-label metrics
- Compute overall multi-label metrics
- Save evaluation reports

---

# Outputs

Running the project generates the following files:

```
outputs/
```

### test_predictions.csv

Contains:

- Original comment
- Ground truth labels
- Predicted labels
- Prediction probabilities

---

### per_label_metrics.csv

Contains:

- Precision
- Recall
- F1-score
- ROC-AUC

for each toxicity category.

---

### overall_metrics.csv

Contains:

- Macro F1 Score
- Micro F1 Score
- Mean ROC-AUC
- Hamming Loss
- Subset Accuracy

---

# Multi-Label Classification

Each comment may belong to multiple toxicity categories simultaneously.

The model uses the **One-vs-Rest** strategy, where six independent Logistic Regression classifiers are trained—one for each toxicity label.

Each classifier predicts its corresponding label independently, making the approach suitable for multi-label classification.

---

# Handling Class Imbalance

The dataset is highly imbalanced, with categories such as **Threat** and **Identity Hate** containing relatively few positive examples.

To address this issue, Logistic Regression is configured with:

```
class_weight="balanced"
```

This automatically adjusts class weights during training to improve learning on minority classes.

---

# Evaluation Metrics

The following metrics are reported.

Per-label:

- Precision
- Recall
- F1-score
- ROC-AUC

Aggregate:

- Macro F1
- Micro F1
- Mean ROC-AUC
- Hamming Loss
- Subset Accuracy

---

# Reproducibility

The project is fully reproducible through:

- Fixed random seed (`random_state = 42`)
- Deterministic train/validation/test split
- Documented preprocessing steps
- Fixed TF-IDF configuration
- Listed software dependencies
- End-to-end executable source code