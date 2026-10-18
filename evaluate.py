"""
Evaluation Module
-----------------
This module:
1. Trains the model
2. Evaluates on the test set
3. Calculates multi-label evaluation metrics
4. Saves the evaluation report
"""

import os
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    hamming_loss,
    accuracy_score
)

from train import train_pipeline


# ======================================================
# Label Names
# ======================================================

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ======================================================
# Evaluation
# ======================================================

def evaluate():

    results = train_pipeline()

    y_true = results["y_test"]
    y_pred = results["y_test_pred"]
    y_prob = results["y_test_prob"]

    evaluation = []

    # ---------------------------------------------
    # Per-label Metrics
    # ---------------------------------------------
    for i, label in enumerate(LABEL_COLUMNS):

        precision = precision_score(
            y_true.iloc[:, i],
            y_pred[:, i],
            zero_division=0
        )

        recall = recall_score(
            y_true.iloc[:, i],
            y_pred[:, i],
            zero_division=0
        )

        f1 = f1_score(
            y_true.iloc[:, i],
            y_pred[:, i],
            zero_division=0
        )

        auc = roc_auc_score(
            y_true.iloc[:, i],
            y_prob[:, i]
        )

        evaluation.append({
            "Label": label,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

    evaluation_df = pd.DataFrame(evaluation)

    # ---------------------------------------------
    # Overall Metrics
    # ---------------------------------------------
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    micro_f1 = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0
    )

    mean_auc = roc_auc_score(
        y_true,
        y_prob,
        average="macro"
    )

    hamming = hamming_loss(
        y_true,
        y_pred
    )

    subset_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # ---------------------------------------------
    # Print Results
    # ---------------------------------------------
    print("\n======================================")
    print("Per Label Evaluation")
    print("======================================")

    print(evaluation_df)

    print("\n======================================")
    print("Overall Metrics")
    print("======================================")

    print(f"Macro F1 Score      : {macro_f1:.4f}")
    print(f"Micro F1 Score      : {micro_f1:.4f}")
    print(f"Mean ROC-AUC        : {mean_auc:.4f}")
    print(f"Hamming Loss        : {hamming:.4f}")
    print(f"Subset Accuracy     : {subset_accuracy:.4f}")

    # ---------------------------------------------
    # Save Results
    # ---------------------------------------------
    os.makedirs("outputs", exist_ok=True)

    evaluation_df.to_csv(
        "outputs/per_label_metrics.csv",
        index=False
    )

    overall_df = pd.DataFrame({
        "Metric": [
            "Macro F1",
            "Micro F1",
            "Mean ROC-AUC",
            "Hamming Loss",
            "Subset Accuracy"
        ],
        "Value": [
            macro_f1,
            micro_f1,
            mean_auc,
            hamming,
            subset_accuracy
        ]
    })

    overall_df.to_csv(
        "outputs/overall_metrics.csv",
        index=False
    )

    print("\nEvaluation reports saved in outputs/")


# ======================================================
# Run
# ======================================================

if __name__ == "__main__":
    evaluate()