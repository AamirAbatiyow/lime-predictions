"""
XGBoost + LIME analysis for Iris dataset splits.

For each JSON dataset:
  - train an XGBClassifier on all rows except the last
  - test on the held-out last row
  - report average accuracy and feature importances
  - save LIME explanation plots for incorrect predictions
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JSON = ROOT / "data" / "processed" / "iris_datasets_15.json"
DEFAULT_LIME_DIR = ROOT / "outputs" / "xgb" / "lime_plots"
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def train_and_test_from_json(json_file: Path, drop_features: list[str] | None = None):
    """Train one XGBClassifier per JSON dataset; test on the last row."""
    with open(json_file, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    le = LabelEncoder()
    all_preds, all_actuals = [], []
    results = []
    feature_importances = []
    feature_columns = None

    for dataset_name, ds in datasets.items():
        df = pd.DataFrame(ds)

        if drop_features:
            df = df.drop(columns=drop_features, errors="ignore")

        X = df.drop(columns=["Id", "Species"])
        y = le.fit_transform(df["Species"])
        feature_columns = X.columns

        X_train, X_test = X.iloc[:-1], X.iloc[-1:]
        y_train, y_test = y[:-1], y[-1:]

        model = XGBClassifier(eval_metric="mlogloss", use_label_encoder=False)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = int(y_pred[0] == y_test[0])

        all_preds.append(y_pred[0])
        all_actuals.append(y_test[0])
        feature_importances.append(model.feature_importances_)

        results.append(
            {
                "dataset_name": dataset_name,
                "X_train": X_train,
                "X_test": X_test,
                "y_test": y_test,
                "y_pred": y_pred,
                "model": model,
                "accuracy": acc,
            }
        )

    avg_acc = sum(p == a for p, a in zip(all_preds, all_actuals)) / len(all_preds)
    print(f"\nAverage accuracy across {len(datasets)} datasets: {avg_acc:.2f}")

    return results, pd.DataFrame(feature_importances, columns=feature_columns)


def save_lime_plot(
    explanation,
    dataset_name: str,
    pred_label: str,
    actual_label: str,
    save_dir: Path,
) -> None:
    """Save a LIME bar chart for one misclassified instance."""
    features, contributions = zip(*explanation.as_list())
    colors = ["green" if c > 0 else "red" for c in contributions]

    plt.figure(figsize=(10, 6))
    plt.barh(features, contributions, color=colors)
    plt.xlabel("Contribution to Prediction")
    plt.title(f"{dataset_name}\nPredicted: {pred_label} | Actual: {actual_label}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{dataset_name}_lime.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved LIME plot: {out_path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run XGBoost + LIME Iris analysis.")
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to processed Iris JSON datasets",
    )
    parser.add_argument(
        "--lime-dir",
        type=Path,
        default=DEFAULT_LIME_DIR,
        help="Directory for LIME plot output",
    )
    parser.add_argument(
        "--drop-features",
        nargs="*",
        default=["SepalWidthCm"],
        help="Feature columns to drop before training",
    )
    parser.add_argument(
        "--keep-old-plots",
        action="store_true",
        help="Do not clear the LIME output directory before running",
    )
    args = parser.parse_args()

    if not args.keep_old_plots and args.lime_dir.exists():
        shutil.rmtree(args.lime_dir)
    args.lime_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing LIME plots to {args.lime_dir}\n")

    results, feature_importances_df = train_and_test_from_json(
        args.json, drop_features=args.drop_features
    )

    avg_importances = feature_importances_df.mean().sort_values()
    print("\nAverage feature importances (low -> high):")
    print(avg_importances)

    for r in results:
        if r["accuracy"] != 0:
            continue

        dataset_name = r["dataset_name"]
        X_train, X_test = r["X_train"], r["X_test"]
        y_test, y_pred = r["y_test"], r["y_pred"]
        model = r["model"]

        print(
            f"\nIncorrect prediction in {dataset_name}: "
            f"Predicted={CLASS_NAMES[y_pred[0]]}, Actual={CLASS_NAMES[y_test[0]]}"
        )

        explainer = LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=X_train.columns.tolist(),
            class_names=CLASS_NAMES,
            mode="classification",
        )

        exp = explainer.explain_instance(
            data_row=X_test.values[0],
            predict_fn=model.predict_proba,
        )

        save_lime_plot(
            explanation=exp,
            dataset_name=dataset_name,
            pred_label=CLASS_NAMES[y_pred[0]],
            actual_label=CLASS_NAMES[y_test[0]],
            save_dir=args.lime_dir,
        )

    print("\nLIME analysis complete.")


if __name__ == "__main__":
    main()
