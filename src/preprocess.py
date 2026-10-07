import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


def normalize_text(text):
    """Normalize text while preserving punctuation, symbols, and casing."""
    if text is None:
        return ""

    # Convert to Unicode NFKC form for consistent character representation.
    text = unicodedata.normalize("NFKC", str(text))

    # Replace curly quotes and long dashes with plain ASCII equivalents.
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Replace line breaks/tabs with spaces and collapse repeated whitespace.
    text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_df(df):
    """Apply text normalization to a dataframe and drop empty rows."""
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")

    normalized_df = df.copy()
    normalized_df["text"] = normalized_df["text"].apply(normalize_text)
    normalized_df = normalized_df[normalized_df["text"].str.len() > 0].reset_index(drop=True)

    return normalized_df


def find_near_duplicate_pairs(df, model_name="paraphrase-multilingual-MiniLM-L12-v2", threshold=0.85):
    """Compute sentence similarity for each qualifying pair and return a dataframe of pair scores."""
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'intent' columns.")

    if df.empty:
        return pd.DataFrame(columns=["left_index", "right_index", "left_text", "right_text", "intent_left", "intent_right", "similarity", "same_intent"])

    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["text"].tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    similarities = cosine_similarity(embeddings)

    upper_mask = np.triu(np.ones_like(similarities, dtype=bool), k=1)
    index_mask = upper_mask & (similarities >= threshold)
    left_idx, right_idx = np.where(index_mask)

    texts = df["text"].tolist()
    intents = df["intent"].tolist()
    rows = []

    for left_pos, right_pos in zip(left_idx.tolist(), right_idx.tolist()):
        left_text = texts[left_pos]
        right_text = texts[right_pos]
        intent_left = intents[left_pos]
        intent_right = intents[right_pos]
        sim = float(similarities[left_pos, right_pos])

        rows.append(
            {
                "left_index": int(left_pos),
                "right_index": int(right_pos),
                "left_text": left_text,
                "right_text": right_text,
                "intent_left": intent_left,
                "intent_right": intent_right,
                "similarity": sim,
                "same_intent": intent_left == intent_right,
            }
        )

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df

    return pair_df.sort_values("similarity", ascending=False).reset_index(drop=True)


def print_near_duplicate_summary(pair_df, top_n=20):
    """Print the most similar pairs and the counts above each similarity threshold."""
    if pair_df.empty:
        print(f"Top {top_n} most similar pairs:")
        print("No near-duplicate pairs found.")
        return

    print(f"Top {top_n} most similar pairs:")
    print(pair_df.head(top_n)[["left_index", "right_index", "similarity", "same_intent", "intent_left", "intent_right"]].to_string(index=False))

    for threshold in [0.85, 0.90, 0.95]:
        same_intent_count = int(((pair_df["similarity"] >= threshold) & (pair_df["same_intent"])).sum())
        diff_intent_count = int(((pair_df["similarity"] >= threshold) & (~pair_df["same_intent"])).sum())
        print(f"Pairs above {threshold}: same-intent={same_intent_count}, different-intent={diff_intent_count}")


def remove_same_intent_duplicates(df, pair_df, similarity_threshold=0.95):
    """Remove near-duplicate rows within the same intent, keeping the first row of each pair."""
    duplicate_rows = set()

    same_intent_pairs = pair_df[(pair_df["same_intent"]) & (pair_df["similarity"] >= similarity_threshold)]

    for _, row in same_intent_pairs.iterrows():
        # Keep the earlier item in the original dataset and drop the later item.
        if row["left_index"] < row["right_index"]:
            duplicate_rows.add(int(row["right_index"]))
        else:
            duplicate_rows.add(int(row["left_index"]))

    cleaned_df = df.drop(index=sorted(duplicate_rows)).reset_index(drop=True)
    return cleaned_df


def save_cross_intent_pairs(pair_df, results_dir="results", output_filename="cross_intent_similar_pairs.csv"):
    """Save different-intent pairs above the similarity threshold for manual review."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cross_intent_pairs = pair_df[(~pair_df["same_intent"]) & (pair_df["similarity"] >= 0.90)].copy()
    if cross_intent_pairs.empty:
        cross_intent_pairs = pd.DataFrame(columns=["left_index", "right_index", "left_text", "right_text", "intent_left", "intent_right", "similarity", "same_intent"])

    cross_intent_pairs = cross_intent_pairs.sort_values("similarity", ascending=False).reset_index(drop=True)
    cross_intent_pairs.to_csv(out_dir / output_filename, index=False)
    return cross_intent_pairs


def remove_near_duplicates(df, model_name="paraphrase-multilingual-MiniLM-L12-v2", threshold=0.85, processed_path="data/processed/synthetic_clean.csv", results_dir="results"):
    """Run the near-duplicate detection pipeline and save the cleaned dataset."""
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'intent' columns.")

    pair_df = find_near_duplicate_pairs(df, model_name=model_name, threshold=threshold)
    print_near_duplicate_summary(pair_df)

    cleaned_df = remove_same_intent_duplicates(df, pair_df, similarity_threshold=0.95)
    save_cross_intent_pairs(pair_df, results_dir=results_dir)

    processed_dir = Path(processed_path).parent
    processed_dir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)

    print(f"Saved cleaned dataset to {processed_path} with {len(cleaned_df)} rows.")
    return cleaned_df


def stratified_split_df(df, train_size=0.70, val_size=0.15, test_size=0.15, random_state=42):
    """Split a dataframe into stratified train, validation, and test sets."""
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("DataFrame must contain 'text' and 'intent' columns.")

    total = train_size + val_size + test_size
    if not np.isclose(total, 1.0):
        raise ValueError("Train, validation, and test sizes must sum to 1.0.")

    df = df.reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df["intent"],
        random_state=random_state,
        shuffle=True,
    )

    val_fraction = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_fraction,
        stratify=temp_df["intent"],
        random_state=random_state,
        shuffle=True,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_counts(train_df, val_df, test_df):
    """Print the count of each intent in each split."""
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"\n{name} split intent counts:")
        print(split_df["intent"].value_counts().sort_index())


def save_splits(df, train_path="data/processed/train.csv", val_path="data/processed/val.csv", test_path="data/processed/test.csv", random_state=42):
    """Save stratified train/validation/test CSV files and return the splits."""
    train_df, val_df, test_df = stratified_split_df(df, random_state=random_state)

    for file_path, split_df in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
        out_dir = Path(file_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(file_path, index=False)

    print_split_counts(train_df, val_df, test_df)
    print(f"Saved train: {train_path}")
    print(f"Saved val: {val_path}")
    print(f"Saved test: {test_path}")

    return train_df, val_df, test_df


def save_data_funnel(raw_df, normalized_df, cleaned_df, train_df, val_df, test_df, output_path="results/data_funnel.csv"):
    """Save a summary table of the number of rows after each preprocessing stage."""
    funnel_df = pd.DataFrame(
        {
            "stage": [
                "raw",
                "after_normalization",
                "after_near_duplicate_removal",
                "train",
                "val",
                "test",
            ],
            "rows": [
                len(raw_df),
                len(normalized_df),
                len(cleaned_df),
                len(train_df),
                len(val_df),
                len(test_df),
            ],
        }
    )

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    funnel_df.to_csv(output_path, index=False)

    print(f"Saved data funnel to {output_path}")
    print(funnel_df.to_string(index=False))
    return funnel_df
