"""
Model Training Module
---------------------
This module:
1. Loads preprocessed data
2. Builds a One-vs-Rest Logistic Regression model
3. Trains the model
4. Generates predictions and prediction probabilities
"""

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import os
import pandas as pd

from preprocessing import preprocess_data


# ======================================================
# Configuration
# ======================================================

RANDOM_STATE = 42

DATA_PATH = "train.csv"


# ======================================================
# Build Model
# ======================================================

def build_model():
    """
    Build One-vs-Rest Logistic Regression model.
    """

    model = OneVsRestClassifier(

        LogisticRegression(

            class_weight="balanced",

            max_iter=1000,

            random_state=RANDOM_STATE

        )

    )

    return model


# ======================================================
# Train Model
# ======================================================

def train_model(model, X_train, y_train):
    """
    Train the model.
    """

    model.fit(X_train, y_train)

    return model


# ======================================================
# Generate Predictions
# ======================================================

def predict(model, X):

    """
    Generate predicted labels and probabilities.
    """

    y_pred = model.predict(X)

    y_prob = model.predict_proba(X)

    return y_pred, y_prob


# ======================================================
# Complete Training Pipeline
# ======================================================

def train_pipeline(file_path=DATA_PATH):

    """
    Complete training workflow.
    """

    (
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
) = preprocess_data(file_path)

    model = build_model()

    model = train_model(
        model,
        X_train,
        y_train
    )

    y_train_pred, y_train_prob = predict(
        model,
        X_train
    )

    y_val_pred, y_val_prob = predict(
        model,
        X_val
    )

    y_test_pred, y_test_prob = predict(
        model,
        X_test
    )

    return {

        "model": model,

        "vectorizer": vectorizer,

        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,

        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,

        "y_train_pred": y_train_pred,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,

        "y_train_prob": y_train_prob,
        "y_val_prob": y_val_prob,
        "y_test_prob": y_test_prob,

        "X_train_text": X_train_text,
        "X_val_text": X_val_text,
        "X_test_text": X_test_text,

    }


# ======================================================
# Run Independently
# ======================================================

if __name__ == "__main__":

    results = train_pipeline()

    print("===================================")
    print("Model Training Completed")
    print("===================================")

    print()

    print("Training samples :", results["X_train"].shape[0])
    print("Validation samples :", results["X_val"].shape[0])
    print("Testing samples :", results["X_test"].shape[0])

    print()

    print("Number of TF-IDF Features :", results["X_train"].shape[1])

    print()

    print("Model Ready for Evaluation")


    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Save validation predictions
    validation_predictions = pd.DataFrame({
        "comment_text": results["X_test_text"]
    }).reset_index(drop=True)

    validation_predictions = pd.concat(
        [
            validation_predictions,
            results["y_test"].reset_index(drop=True)
        ],
        axis=1
    )

    label_columns = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]

    for i, col in enumerate(label_columns):
        validation_predictions[f"{col}_pred"] = results["y_test_pred"][:, i]
        validation_predictions[f"{col}_prob"] = results["y_test_prob"][:, i]

    validation_predictions.to_csv(
        "outputs/test_predictions.csv",
        index=False
    )

    print("\nTest predictions saved to outputs/test_predictions.csv")