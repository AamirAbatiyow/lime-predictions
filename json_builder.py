import pandas as pd
import random
import json

def split_iris_into_datasets(csv_path="iris.csv", output_json="iris_datasets.json"):
    # Load CSV
    df = pd.read_csv(csv_path)

    # Separate by species
    setosa = df[df['Species'] == 'Iris-setosa'].sample(frac=1, random_state=42).reset_index(drop=True)
    versicolor = df[df['Species'] == 'Iris-versicolor'].sample(frac=1, random_state=43).reset_index(drop=True)
    virginica = df[df['Species'] == 'Iris-virginica'].sample(frac=1, random_state=44).reset_index(drop=True)

    datasets = []
    used_setosa = used_versicolor = used_virginica = 0

    for i in range(15):
        # Get 3 samples from each species
        s_part = setosa.iloc[used_setosa:used_setosa + 3]
        v_part = versicolor.iloc[used_versicolor:used_versicolor + 3]
        g_part = virginica.iloc[used_virginica:used_virginica + 3]

        used_setosa += 3
        used_versicolor += 3
        used_virginica += 3

        # Randomly choose a 10th sample from any species with remaining data
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

        # Combine and shuffle
        dataset = pd.concat([s_part, v_part, g_part, tenth]).sample(frac=1).reset_index(drop=True)
        datasets.append(dataset)

    # Convert to a JSON-serializable dictionary
    datasets_json = {f"dataset_{i+1}": ds.to_dict(orient="records") for i, ds in enumerate(datasets)}

    # Save to JSON file
    with open(output_json, "w") as f:
        json.dump(datasets_json, f, indent=4)

    print(f"✅ Saved {len(datasets)} datasets to {output_json}")

    return datasets

# Example usage
if __name__ == "__main__":
    all_sets = split_iris_into_datasets("iris.csv", "iris_datasets.json")
    for i, ds in enumerate(all_sets, start=1):
        print(f"\nDataset {i}:")
        print(ds[['Id', 'Species']])
