"""
Build Iris JSON dataset splits for model training and LIME analysis.

Reads the raw Iris CSV, shuffles samples within each species, and writes
balanced multi-dataset JSON files used by the analysis scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data" / "raw" / "iris.csv"
DEFAULT_OUT = ROOT / "data" / "processed" / "iris_datasets.json"


def split_iris_into_datasets(
    csv_path: Path,
    output_json: Path,
    num_datasets: int = 5,
    samples_per_species: int = 10,
) -> list[pd.DataFrame]:
    """Split the Iris CSV into balanced, shuffled JSON datasets."""
    df = pd.read_csv(csv_path)

    setosa = (
        df[df["Species"] == "Iris-setosa"]
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    versicolor = (
        df[df["Species"] == "Iris-versicolor"]
        .sample(frac=1, random_state=43)
        .reset_index(drop=True)
    )
    virginica = (
        df[df["Species"] == "Iris-virginica"]
        .sample(frac=1, random_state=44)
        .reset_index(drop=True)
    )

    needed = num_datasets * samples_per_species
    for name, species_df in (
        ("Iris-setosa", setosa),
        ("Iris-versicolor", versicolor),
        ("Iris-virginica", virginica),
    ):
        if len(species_df) < needed:
            raise ValueError(
                f"Need {needed} rows of {name}, but only found {len(species_df)}."
            )

    datasets: list[pd.DataFrame] = []
    used = 0
    for _ in range(num_datasets):
        start, end = used, used + samples_per_species
        dataset = (
            pd.concat(
                [
                    setosa.iloc[start:end],
                    versicolor.iloc[start:end],
                    virginica.iloc[start:end],
                ]
            )
            .sample(frac=1, random_state=used + 1)
            .reset_index(drop=True)
        )
        datasets.append(dataset)
        used = end

    datasets_json = {
        f"dataset_{i + 1}": ds.to_dict(orient="records")
        for i, ds in enumerate(datasets)
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(datasets_json, f, indent=4)

    print(f"Saved {len(datasets)} datasets to {output_json}")
    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Iris JSON dataset splits.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Path to Iris CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=5,
        help="Number of dataset splits to create",
    )
    parser.add_argument(
        "--samples-per-species",
        type=int,
        default=10,
        help="Rows per species in each dataset",
    )
    args = parser.parse_args()

    all_sets = split_iris_into_datasets(
        csv_path=args.csv,
        output_json=args.output,
        num_datasets=args.num_datasets,
        samples_per_species=args.samples_per_species,
    )
    for i, ds in enumerate(all_sets, start=1):
        print(f"\nDataset {i}:")
        print(ds[["Id", "Species"]])


if __name__ == "__main__":
    main()
