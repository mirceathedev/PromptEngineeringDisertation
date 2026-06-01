import os
import pandas as pd
from datasets import load_dataset

# Ensure target directory exists
os.makedirs("datasets", exist_ok=True)


# -----------------------------
# GENERIC HELPERS
# -----------------------------
def load_dataset_flexible(repo_name, split_preferences=("train", "test", "validation", "original")):
    """
    Loads a Hugging Face dataset safely.
    Works with Dataset or DatasetDict.
    """
    dataset = load_dataset(repo_name)

    if hasattr(dataset, "keys"):
        available_splits = list(dataset.keys())
        print(f"Available splits for {repo_name}: {available_splits}")

        for split_name in split_preferences:
            if split_name in dataset:
                return dataset[split_name]

        # fallback: first available split
        return dataset[available_splits[0]]

    return dataset


def find_text_column(df):
    candidate_columns = [
        "text",
        "sentence",
        "question",
        "utterance",
        "utt",
        "query",
        "content",
        "title",
        "description"
    ]

    for col in candidate_columns:
        if col in df.columns:
            return col

    # fallback: first object/string column
    for col in df.columns:
        if df[col].dtype == "object":
            return col

    raise ValueError(f"Could not find text column. Columns: {df.columns.tolist()}")


def find_label_column(df):
    candidate_columns = [
        "label",
        "labels",
        "category",
        "class",
        "intent",
        "target",
        "sport",
        "topic"
    ]

    for col in candidate_columns:
        if col in df.columns:
            return col

    raise ValueError(f"Could not find label column. Columns: {df.columns.tolist()}")


def build_label_map_from_features(raw_dataset, label_column):
    """
    If Hugging Face ClassLabel metadata exists, use it.
    Otherwise return None.
    """
    try:
        feature = raw_dataset.features[label_column]
        if hasattr(feature, "names") and feature.names:
            return {i: name.upper().replace("_", " ") for i, name in enumerate(feature.names)}
    except Exception:
        pass

    return None


def standardize_dataset(raw_dataset, dataset_name):
    df = pd.DataFrame(raw_dataset)

    print(f"\nColumns for {dataset_name}: {df.columns.tolist()}")

    text_col = find_text_column(df)
    label_col = find_label_column(df)

    label_map = build_label_map_from_features(raw_dataset, label_col)

    df = df.rename(columns={text_col: "review_text"})

    if label_map and pd.api.types.is_numeric_dtype(df[label_col]):
        df["ground_truth"] = df[label_col].map(label_map)
    else:
        df["ground_truth"] = (
            df[label_col]
            .astype(str)
            .str.upper()
            .str.replace("_", " ", regex=False)
            .str.strip()
        )

    # Keep only useful columns first, but preserve original columns too
    front_cols = ["review_text", "ground_truth"]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + other_cols]

    # Remove broken rows
    df = df.dropna(subset=["review_text", "ground_truth"])
    df = df[df["review_text"].astype(str).str.len() > 0]
    df = df[df["ground_truth"].astype(str).str.len() > 0]

    return df


def balanced_sample(df, label_column="ground_truth", examples_per_class=100, random_state=42):
    subsets = []

    labels = sorted(df[label_column].dropna().unique())

    for label in labels:
        subset = df[df[label_column] == label]

        if len(subset) < examples_per_class:
            print(
                f"⚠️ Skipping class '{label}' because it has only "
                f"{len(subset)} rows, less than requested {examples_per_class}."
            )
            continue

        sampled_subset = subset.sample(
            n=examples_per_class,
            random_state=random_state
        )

        subsets.append(sampled_subset)

    if not subsets:
        raise ValueError("No class had enough examples. Try a smaller examples_per_class value.")

    sampled_df = (
        pd.concat(subsets)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    return sampled_df


def save_dataset(df, output_filename, examples_per_class):
    output_path = os.path.join("datasets", output_filename)
    df.to_excel(output_path, index=False)

    class_counts = df["ground_truth"].value_counts().sort_index()

    print(f"\n✅ Dataset saved to {output_path}")
    print(f"   Total rows: {len(df)}")
    print(f"   Requested: {examples_per_class} per class")
    print("\nClass distribution:")
    print(class_counts)


# -----------------------------
# SST-5 DATASET
# -----------------------------
def prepare_sst5(examples_per_class=100):
    print("\n⏳ Loading SST-5 Dataset from Hugging Face...")

    raw_dataset = load_dataset("SetFit/sst5", split="train")
    df = pd.DataFrame(raw_dataset)

    label_map = {
        0: "VERY NEGATIVE",
        1: "NEGATIVE",
        2: "NEUTRAL",
        3: "POSITIVE",
        4: "VERY POSITIVE"
    }

    df["ground_truth"] = df["label"].map(label_map)

    if "text" in df.columns:
        df = df.rename(columns={"text": "review_text"})

    sampled_df = balanced_sample(
        df=df,
        label_column="ground_truth",
        examples_per_class=examples_per_class
    )

    save_dataset(sampled_df, "dataset_sst5.xlsx", examples_per_class)


# -----------------------------
# SPORTS TEXT CLASSIFICATION
# -----------------------------
def prepare_sports(examples_per_class=100):
    print("\n⏳ Loading Sports Text Classification Dataset from Hugging Face...")

    # This dataset has columns: football, basketball, label
    # So we manually convert it from wide format to normal format.
    raw_dataset = load_dataset(
        "its-zion-18/sports-text-dataset",
        split="original"
    )

    df_wide = pd.DataFrame(raw_dataset)

    print("\nSports raw columns:", df_wide.columns.tolist())

    rows = []

    for _, row in df_wide.iterrows():
        if "football" in df_wide.columns and pd.notna(row["football"]):
            rows.append({
                "review_text": str(row["football"]),
                "ground_truth": "FOOTBALL"
            })

        if "basketball" in df_wide.columns and pd.notna(row["basketball"]):
            rows.append({
                "review_text": str(row["basketball"]),
                "ground_truth": "BASKETBALL"
            })

    df = pd.DataFrame(rows)

    sampled_df = balanced_sample(
        df=df,
        label_column="ground_truth",
        examples_per_class=examples_per_class
    )

    save_dataset(sampled_df, "dataset_sports.xlsx", examples_per_class)


# -----------------------------
# COMMAND / INTENT CLASSIFICATION
# -----------------------------
def prepare_commands(examples_per_class=100, max_classes=10):
    print("\n⏳ Loading Commands / Intent Classification Dataset from Hugging Face...")

    # Parquet mirror of BANKING77.
    # Treat each user query as a command/intent classification example.
    raw_dataset = load_dataset_flexible("gtfintechlab/banking77")

    df = standardize_dataset(
        raw_dataset=raw_dataset,
        dataset_name="banking77-commands"
    )

    print("\nDetected command/intent classes:")
    print(df["ground_truth"].value_counts())

    # BANKING77 has 77 classes. For prompt experiments, using all 77 is too hard.
    # Keep only the largest max_classes classes, 100 examples each.
    top_labels = (
        df["ground_truth"]
        .value_counts()
        .head(max_classes)
        .index
        .tolist()
    )

    df = df[df["ground_truth"].isin(top_labels)].reset_index(drop=True)

    sampled_df = balanced_sample(
        df=df,
        label_column="ground_truth",
        examples_per_class=examples_per_class
    )

    save_dataset(sampled_df, "dataset_commands.xlsx", examples_per_class)


# -----------------------------
# AG NEWS TOPIC CLASSIFICATION
# -----------------------------
def prepare_agnews(examples_per_class=100):
    print("\n⏳ Loading AG News Topic Classification Dataset from Hugging Face...")

    raw_dataset = load_dataset("ag_news", split="train")
    df = pd.DataFrame(raw_dataset)

    label_names = raw_dataset.features["label"].names
    label_map = {
        i: label_names[i].upper().replace("/", " / ")
        for i in range(len(label_names))
    }

    df["ground_truth"] = df["label"].map(label_map)

    if "text" in df.columns:
        df = df.rename(columns={"text": "review_text"})

    sampled_df = balanced_sample(
        df=df,
        label_column="ground_truth",
        examples_per_class=examples_per_class
    )

    save_dataset(sampled_df, "dataset_agnews.xlsx", examples_per_class)


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    examples_per_class = 100

    prepare_sst5(examples_per_class=examples_per_class)
    prepare_agnews(examples_per_class=examples_per_class)

    # New requested datasets
    prepare_sports(examples_per_class=50)
    prepare_commands(examples_per_class=examples_per_class, max_classes=10)

    print("\n🎉 Setup Complete!")
    print("Generated datasets:")
    print(" - datasets/dataset_sst5.xlsx")
    print(" - datasets/dataset_agnews.xlsx")
    print(" - datasets/dataset_sports.xlsx")
    print(" - datasets/dataset_commands.xlsx")