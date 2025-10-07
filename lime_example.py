
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