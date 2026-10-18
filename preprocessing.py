"""
Preprocessing Module
--------------------
This module:
1. Loads the dataset
2. Cleans comment text
3. Splits data into Train/Validation/Test (70:10:20)
4. Converts text into TF-IDF features
"""

import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ======================================================
# Configuration
# ======================================================

RANDOM_STATE = 42

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ======================================================
# Load Dataset
# ======================================================

def load_data(file_path):
    """
    Load CSV dataset.
    """

    df = pd.read_csv(file_path)

# Keep only required columns
    df = df[
        [
            "comment_text",
            "toxic",
            "severe_toxic",
            "obscene",
            "threat",
            "insult",
            "identity_hate",
        ]
    ]

    return df


# ======================================================
# Text Cleaning
# ======================================================

def clean_text(text):
    """
    Clean raw comment text.

    Steps:
    - Convert to lowercase
    - Remove URLs
    - Remove HTML tags
    - Replace newline/tab with space
    - Remove extra whitespace
    """

    text = str(text)

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", " ", text)

    # remove html tags
    text = re.sub(r"<.*?>", " ", text)

    # remove newline/tab
    text = re.sub(r"[\n\t\r]", " ", text)

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ======================================================
# Prepare Dataset
# ======================================================

def prepare_dataset(df):
    """
    Prepare text and labels.
    """

    # Clean comments
    df["comment_text"] = df["comment_text"].apply(clean_text)

    # Features
    X = df["comment_text"]

    # Targets (multi-label)
    y = df[LABEL_COLUMNS]

    return X, y


# ======================================================
# Train / Validation / Test Split
# ======================================================

def split_data(X, y):
    """
    Split dataset into

    Train : 70%
    Validation : 10%
    Test : 20%
    """

    # 80% train+val
    # 20% test

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # From remaining 80%
    # 70 train
    # 10 validation

    validation_ratio = 0.125

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )


# ======================================================
# TF-IDF Vectorization
# ======================================================

def vectorize_text(
    X_train,
    X_val,
    X_test
):
    """
    Convert text into TF-IDF vectors.
    """

    vectorizer = TfidfVectorizer(

        lowercase=False,

        ngram_range=(1, 2),

        max_features=50000,

        min_df=2

    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    X_val_tfidf = vectorizer.transform(X_val)

    X_test_tfidf = vectorizer.transform(X_test)

    return (
        X_train_tfidf,
        X_val_tfidf,
        X_test_tfidf,
        vectorizer
    )


# ======================================================
# Complete Pipeline
# ======================================================

def preprocess_data(file_path):
    """
    Complete preprocessing pipeline.
    """

    df = load_data(file_path)

    X, y = prepare_dataset(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    ) = split_data(X, y)

    X_train_text = X_train.copy()
    X_val_text = X_val.copy()
    X_test_text = X_test.copy()

    (
        X_train,
        X_val,
        X_test,
        vectorizer
    ) = vectorize_text(
        X_train,
        X_val,
        X_test
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        vectorizer,
        X_train_text,
        X_val_text,
        X_test_text
    )