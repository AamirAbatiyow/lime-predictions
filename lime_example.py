'''
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    mode="classification"
)

i = 3
sample = X_test[i]

print("\nTrue class:", iris.target_names[y_test[i]])
print("Predicted class:", iris.target_names[model.predict([sample])[0]])

exp = explainer.explain_instance(
    data_row=sample,
    predict_fn=model.predict_proba,
    num_features=2
)

print("\nExplanation:")
print(exp.as_list())

for label in exp.available_labels():  # only labels LIME has explanations for
    fig = exp.as_pyplot_figure(label=label)
    plt.title(f"Explanation for class: {iris.target_names[label]}")
    plt.show()


fig = exp.as_pyplot_figure()
plt.show()
'''

import pandas as pd
import random

def split_iris_into_datasets(csv_path="iris.csv"):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Separate by species
    setosa = df[df['Species'] == 'Iris-setosa'].sample(frac=1, random_state=42).reset_index(drop=True)
    versicolor = df[df['Species'] == 'Iris-versicolor'].sample(frac=1, random_state=43).reset_index(drop=True)
    virginica = df[df['Species'] == 'Iris-virginica'].sample(frac=1, random_state=44).reset_index(drop=True)

    datasets = []
    used_setosa = 0
    used_versicolor = 0
    used_virginica = 0

    for i in range(15):
        # Get 3 samples from each species
        s_part = setosa.iloc[used_setosa:used_setosa + 3]
        v_part = versicolor.iloc[used_versicolor:used_versicolor + 3]
        g_part = virginica.iloc[used_virginica:used_virginica + 3]

        used_setosa += 3
        used_versicolor += 3
        used_virginica += 3

        # For the 10th entry: choose randomly among remaining of any species
        remaining_choices = []
        if used_setosa < len(setosa): remaining_choices.append(('setosa', used_setosa))
        if used_versicolor < len(versicolor): remaining_choices.append(('versicolor', used_versicolor))
        if used_virginica < len(virginica): remaining_choices.append(('virginica', used_virginica))

        species_choice, idx = random.choice(remaining_choices)

        if species_choice == 'setosa':
            tenth = setosa.iloc[[idx]]
            used_setosa += 1
        elif species_choice == 'versicolor':
            tenth = versicolor.iloc[[idx]]
            used_versicolor += 1
        else:
            tenth = virginica.iloc[[idx]]
            used_virginica += 1

        # Combine into one dataset and shuffle
        dataset = pd.concat([s_part, v_part, g_part, tenth]).sample(frac=1).reset_index(drop=True)
        datasets.append(dataset)

    return datasets

# Example usage:
if __name__ == "__main__":
    all_sets = split_iris_into_datasets("iris.csv")
    for i, ds in enumerate(all_sets, start=1):
        print(f"\nDataset {i}:")
        print(ds[['Id', 'Species']])
