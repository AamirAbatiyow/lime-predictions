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

    for i in range(5):
        # Get 3 samples from each species
        s_part = setosa.iloc[used_setosa:used_setosa + 10]
        v_part = versicolor.iloc[used_versicolor:used_versicolor + 10]
        g_part = virginica.iloc[used_virginica:used_virginica + 10]

        used_setosa += 10
        used_versicolor += 10
        used_virginica += 10

        # Randomly choose a 10th sample from any species with remaining data
       

        # Combine and shuffle
        dataset = pd.concat([s_part, v_part, g_part]).sample(frac=1).reset_index(drop=True)
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
