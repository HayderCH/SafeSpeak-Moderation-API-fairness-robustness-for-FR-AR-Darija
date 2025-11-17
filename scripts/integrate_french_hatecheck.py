#!/usr/bin/env python3
"""
Integrate French HateCheck dataset to boost French performance
"""
import pandas as pd


def load_french_hatecheck():
    """Load and preprocess French HateCheck dataset"""
    df = pd.read_csv("data/raw/public/hatecheck_cases_final_french.csv")

    # Map HateCheck labels to our format
    label_mapping = {"hateful": "Toxic", "non-hateful": "Neutral"}

    df["label"] = df["label_gold"].map(label_mapping)
    df["language"] = "fr"
    df["text"] = df["test_case"]
    df["source"] = "hatecheck_french"
    df["split"] = "train"  # Will be split later

    # Select relevant columns
    columns = ["text", "label", "language", "source", "split"]
    df = df[columns]

    return df


def integrate_french_data():
    """Integrate French HateCheck data with existing training data"""
    print("Loading existing training data...")
    train_df = pd.read_csv("data/final/train_corrected.csv")

    print("Loading French HateCheck data...")
    french_hc_df = load_french_hatecheck()

    print(f"Original French samples: {(train_df['language']=='fr').sum()}")
    print(f"Additional French samples: {len(french_hc_df)}")

    # Combine datasets
    combined_df = pd.concat([train_df, french_hc_df], ignore_index=True)

    # Shuffle the combined dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total samples after integration: {len(combined_df)}")
    print(f"New French samples: {(combined_df['language']=='fr').sum()}")

    # Save the enhanced dataset
    output_path = "data/final/train_corrected_french_enhanced.csv"
    combined_df.to_csv(output_path, index=False)
    print(f"Saved enhanced dataset to: {output_path}")

    # Show new French distribution
    french_data = combined_df[combined_df["language"] == "fr"]
    print("New French label distribution:")
    print(french_data["label"].value_counts())
    print(".3f")

    return combined_df


if __name__ == "__main__":
    integrate_french_data()
