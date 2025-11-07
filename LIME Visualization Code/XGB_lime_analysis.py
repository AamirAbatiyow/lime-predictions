# ==========================================================
# XGBoost + LIME Analysis Script
# ----------------------------------------------------------
# This script:
#   • Loads multiple datasets (from JSON)
#   • Trains an XGBClassifier on each
#   • Tests on a held-out row
#   • Evaluates accuracy
#   • Uses LIME to visualize feature contributions
#   • Saves LIME plots (only for incorrect predictions)
#   • Clears the 'lime_plots' folder before every run
# ==========================================================

# -----------------------------
# Imports
# -----------------------------
import json
import os
import shutil            # For clearing plot directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer
import warnings

# Silence unimportant warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ==========================================================
# Function: Train & Test on JSON datasets
# ==========================================================
def train_and_test_from_json(json_file, drop_features=None):
    """
    Loads multiple datasets from a JSON file and trains
    an XGBClassifier on each, testing on the last row.

    Returns:
        results (list): Info on models and predictions
        feature_importances_df (DataFrame): Feature importances across datasets
    """
    with open(json_file, "r") as f:
        datasets = json.load(f)

    le = LabelEncoder()
    all_preds, all_actuals = [], []
    results = []
    feature_importances = []

    for dataset_name, ds in datasets.items():
        df = pd.DataFrame(ds)

        # Drop unwanted columns if specified
        if drop_features:
            df = df.drop(columns=drop_features, errors="ignore")

        # Split into features and labels
        X = df.drop(columns=["Id", "Species"])
        y = le.fit_transform(df["Species"])

        # Train on all but last row, test on last row
        X_train, X_test = X.iloc[:-1], X.iloc[-1:]
        y_train, y_test = y[:-1], y[-1:]

        # Train the model
        model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
        model.fit(X_train, y_train)

        # Evaluate on test row
        y_pred = model.predict(X_test)
        acc = int(y_pred[0] == y_test[0])

        # Collect results
        all_preds.append(y_pred[0])
        all_actuals.append(y_test[0])
        feature_importances.append(model.feature_importances_)

        results.append({
            "dataset_name": dataset_name,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "model": model,
            "accuracy": acc
        })

    avg_acc = sum(p == a for p, a in zip(all_preds, all_actuals)) / len(all_preds)
    print(f"\n✅ Average Accuracy across {len(datasets)} datasets: {avg_acc:.2f}")

    return results, pd.DataFrame(feature_importances, columns=X.columns)


# ==========================================================
# Function: Save LIME Explanations
# ==========================================================
def save_lime_plot(explanation, dataset_name, pred_label, actual_label, save_dir="lime_plots"):
    """
    Saves a LIME explanation plot showing which features
    pushed the model prediction toward (green) or away (red)
    from the predicted class.
    """
    features, contributions = zip(*explanation.as_list())
    colors = ["green" if c > 0 else "red" for c in contributions]

    plt.figure(figsize=(10, 6))
    plt.barh(features, contributions, color=colors)
    plt.xlabel("Contribution to Prediction")
    plt.title(f"{dataset_name}\nPredicted: {pred_label} | Actual: {actual_label}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{dataset_name}_lime.png"))
    plt.close()
    print(f"💾 Saved LIME plot: {dataset_name}_lime.png")


# ==========================================================
# MAIN EXECUTION
# ==========================================================
if __name__ == "__main__":
    json_file = "iris_datasets_15.json"
    drop_features = ["SepalWidthCm"]  # Add feature names to remove if testing feature reduction
    lime_dir = "lime_plots"

    # --- Clear lime_plots directory before each run ---
    if os.path.exists(lime_dir):
        shutil.rmtree(lime_dir)
    os.makedirs(lime_dir, exist_ok=True)
    print("🧹 Cleared old LIME plots.\n")

    # --- Train and test ---
    results, feature_importances_df = train_and_test_from_json(json_file, drop_features)

    # --- Show average feature importances ---
    avg_importances = feature_importances_df.mean().sort_values()
    print("\n📊 Average Feature Importances (low → high):")
    print(avg_importances)

    # --- Generate LIME visualizations for incorrect predictions ---
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    for r in results:
        if r["accuracy"] == 0:  # Only visualize incorrect predictions
            dataset_name = r["dataset_name"]
            X_train, X_test = r["X_train"], r["X_test"]
            y_test, y_pred = r["y_test"], r["y_pred"]
            model = r["model"]

            print(f"\n❌ Incorrect Prediction in {dataset_name}: "
                  f"Predicted={class_names[y_pred[0]]}, Actual={class_names[y_test[0]]}")

            # Explain this instance
            explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=class_names,
                mode="classification"
            )

            exp = explainer.explain_instance(
                data_row=X_test.values[0],
                predict_fn=model.predict_proba
            )

            save_lime_plot(
                explanation=exp,
                dataset_name=dataset_name,
                pred_label=class_names[y_pred[0]],
                actual_label=class_names[y_test[0]],
                save_dir=lime_dir
            )

    print("\n✅ LIME analysis complete — incorrect predictions plotted and saved.")
