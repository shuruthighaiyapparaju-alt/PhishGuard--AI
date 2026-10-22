"""
train_model.py
--------------
Trains and evaluates the phishing email classifier.

Run it from the project folder:
    python train_model.py

What it does:
  1. Loads data/emails.csv
  2. Removes missing values and duplicates
  3. Splits the data into training (80%) and testing (20%) sets
  4. Trains the Pipeline (TF-IDF + URL features + Logistic Regression)
  5. Evaluates on the unseen test set
  6. Saves: models/phishing_model.joblib, results/confusion_matrix.png,
            results/class_distribution.png, results/metrics.json
"""

import argparse
import json
import sys
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")  # draw plots without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split

from preprocess import (CLASS_DIST_PATH, CONFUSION_MATRIX_PATH, DATA_PATH,
                        LABEL_NAMES, METRICS_PATH, MODEL_PATH, RESULTS_DIR,
                        build_pipeline, clean_dataframe, load_dataset)


def plot_class_distribution(labels):
    counts = labels.map(LABEL_NAMES).value_counts()
    plt.figure(figsize=(5, 4))
    ax = sns.barplot(x=counts.index, y=counts.values,
                     hue=counts.index, palette="Set2", legend=False)
    for container in ax.containers:
        ax.bar_label(container)
    plt.title("Class distribution (after cleaning)")
    plt.ylabel("Number of emails")
    plt.tight_layout()
    plt.savefig(CLASS_DIST_PATH, dpi=150)
    plt.close()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Safe", "Phishing"],
                yticklabels=["Safe", "Phishing"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title("Confusion matrix (unseen test data)")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train phishing detector")
    parser.add_argument("--data", default=str(DATA_PATH),
                        help="path to the CSV file (default: data/emails.csv)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="use only the first N rows (for a quick test)")
    args = parser.parse_args()

    # ---- 1. Load and clean -------------------------------------------------
    print("Step 1/5: Loading dataset ...")
    try:
        df = load_dataset(args.data, max_rows=args.max_rows)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nERROR: {error}")
        sys.exit(1)

    df, stats = clean_dataframe(df)
    print(f"  Rows loaded:                {stats['rows_loaded']}")
    print(f"  After removing empty rows:  {stats['rows_after_empty_removed']}")
    print(f"  After removing duplicates:  "
          f"{stats['rows_after_duplicates_removed']}")

    class_counts = df["label"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 10:
        print("\nERROR: The dataset needs at least 10 emails of EACH class "
              "(Safe and Phishing) after cleaning.")
        sys.exit(1)

    minority_share = class_counts.min() / len(df)
    print(f"  Safe: {class_counts.get(0, 0)}   "
          f"Phishing: {class_counts.get(1, 0)}")
    if minority_share < 0.3:
        print("  NOTE: classes are imbalanced - do not rely on accuracy "
              "alone. class_weight='balanced' is used.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    plot_class_distribution(df["label"])

    # ---- 2. Split ----------------------------------------------------------
    print("Step 2/5: Splitting into train (80%) and test (20%) ...")
    X = df[["subject", "body"]]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"  Training emails: {len(X_train)}   Test emails: {len(X_test)}")

    # ---- 3. Train ----------------------------------------------------------
    print("Step 3/5: Training the model (this can take a few minutes) ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)  # learns from TRAINING data only

    # ---- 4. Evaluate -------------------------------------------------------
    print("Step 4/5: Evaluating on unseen test data ...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, pos_label=1, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    plot_confusion_matrix(cm)

    print("\n=== RESULTS ON THE TEST SET (Phishing = positive class) ===")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n")
    print(classification_report(y_test, y_pred,
                                target_names=["Safe", "Phishing"],
                                zero_division=0))

    # ---- 5. Save -----------------------------------------------------------
    print("Step 5/5: Saving model and results ...")
    joblib.dump({"pipeline": pipeline,
                 "sklearn_version": sklearn.__version__}, MODEL_PATH)

    metrics = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_file": str(args.data),
        "dataset_stats": stats,
        "class_counts": {"Safe": int(class_counts.get(0, 0)),
                         "Phishing": int(class_counts.get(1, 0))},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Safe", "Phishing"],
            output_dict=True, zero_division=0),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"  Model saved to:            {MODEL_PATH}")
    print(f"  Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")
    print(f"  Metrics saved to:          {METRICS_PATH}")
    print("\nDone! Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
