import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from lime.lime_tabular import LimeTabularExplainer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Function: Train & Test on JSON datasets
# -----------------------------
def train_and_test_from_json(json_file, drop_features=None):
    with open(json_file, "r") as f:
        datasets = json.load(f)

    le = LabelEncoder()
    all_preds, all_actuals = [], []
    results = []
    feature_importances = []

    for i, (dataset_name, ds) in enumerate(datasets.items(), start=1):
        df = pd.DataFrame(ds)

        # Drop any unwanted columns
        if drop_features:
            df = df.drop(columns=drop_features, errors='ignore')

        X = df.drop(columns=["Id", "Species"])
        y = le.fit_transform(df["Species"])

        # Train on all rows except last, test on last row
        X_train, X_test = X.iloc[:-1], X.iloc[-1:]
        y_train, y_test = y[:-1], y[-1:]

        model = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = (y_pred[0] == y_test[0])

        # print(f"{dataset_name}: Predicted = {le.inverse_transform(y_pred)[0]}, Actual = {le.inverse_transform(y_test)[0]}, Accuracy = {acc:.2f}")

        all_preds.append(y_pred[0])
        all_actuals.append(y_test[0])
        feature_importances.append(model.feature_importances_)

        results.append({
            "dataset_name": dataset_name,
            "X_train": X_train,
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "model": model
        })

    avg_acc = sum([p == a for p, a in zip(all_preds, all_actuals)]) / len(datasets)
    print(f"\n✅ Average Accuracy across {len(datasets)} datasets: {avg_acc:.2f}")

    return results, pd.DataFrame(feature_importances, columns=X.columns)


# -----------------------------
# Function: Plot LIME explanations
# -----------------------------
def plot_lime_dual(exp_pred, exp_actual, feature_names, dataset_num, pred_label, actual_label, save_dir="XGB/lime plots"):
    features_pred, contributions_pred = zip(*exp_pred.as_list())
    features_actual, contributions_actual = zip(*exp_actual.as_list())

    x = range(len(features_pred))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.barh([f + " " for f in features_pred], contributions_pred, height=width, color='green', label=f'Predicted ({pred_label})')
    plt.barh([f for f in features_actual], contributions_actual, height=width, color='red', alpha=0.5, label=f'Actual ({actual_label})')
    plt.xlabel("Contribution to prediction")
    plt.title(f"Dataset {dataset_num}: Predicted={pred_label}, Actual={actual_label}")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{dataset_num}_lime_dual.png"))
    plt.close()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    json_file = "iris_datasets_15.json"
    drop_features = []  # drop features you don't want

    # Run training/testing
    results, feature_importances_df = train_and_test_from_json(json_file, drop_features=drop_features)

    # Show average feature importances
    avg_importances = feature_importances_df.mean().sort_values()
    print("\n📊 Average Feature Importances (low -> high):")
    print(avg_importances)

    # Class names for LIME
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

    # LIME explanations for incorrect predictions
    for r in results:
        dataset_name = r["dataset_name"]
        X_train = r["X_train"]
        X_test = r["X_test"]
        y_test = r["y_test"]
        y_pred = r["y_pred"]
        model = r["model"]

        if y_pred[0] != y_test[0]:
            print(f"\n❌ {dataset_name} - Incorrect Prediction")

            explainer = LimeTabularExplainer(
                training_data=X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=class_names,
                mode="classification"
            )

            # Predicted explanation
            exp_pred = explainer.explain_instance(
                data_row=X_test.values[0],
                predict_fn=model.predict_proba
            )

            # Actual explanation: simulate one-hot probabilities for the true class
            def predict_actual_proba(X_input):
                X_input = np.array(X_input)
                return np.array([[1 if i == y_test[0] else 0 for i in range(len(class_names))] for _ in range(X_input.shape[0])])

            exp_actual = explainer.explain_instance(
                data_row=X_test.values[0],
                predict_fn=predict_actual_proba
            )

            plot_lime_dual(
                exp_pred,
                exp_actual,
                feature_names=X_train.columns.tolist(),
                dataset_num=dataset_name,
                pred_label=class_names[y_pred[0]],
                actual_label=class_names[y_test[0]]
            )

            print(f"✅ LIME explanation saved for {dataset_name}")
