import pandas as pd
import os
import re


def parse_conllu_file(filepath):
    """Parse CoNLL-U file and extract text and offensive classification."""
    sentences = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into sentence blocks
    sentence_blocks = re.split(r"\n\s*\n", content.strip())

    for block in sentence_blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        text = None
        offensive = None
        sent_id = None

        for line in lines:
            if line.startswith("# text = "):
                text = line[9:].strip()  # Remove '# text = '
            elif line.startswith("# offensive_classification = "):
                # Remove '# offensive_classification = '
                offensive = int(line[29:].strip())
            elif line.startswith("# sent_id = "):
                sent_id = line[12:].strip()  # Remove '# sent_id = '

        if text and offensive is not None:
            sentences.append({"id": sent_id, "text": text, "offensive": offensive})

    return sentences


# Parse all files
base_path = (
    "data/raw/public/release-narabizi-treebank-master/"
    "release-narabizi-treebank-master"
)
all_sentences = []

files = [
    "qaf_arabizi-ud-train.conllu",
    "qaf_arabizi-ud-dev.conllu",
    "qaf_arabizi-ud-test.conllu",
]

for file in files:
    filepath = os.path.join(base_path, file)
    sentences = parse_conllu_file(filepath)
    print(f"Parsed {len(sentences)} sentences from {file}")

    # Add split info
    if "train" in file:
        split = "train"
    elif "dev" in file:
        split = "validation"
    elif "test" in file:
        split = "test"

    for sent in sentences:
        sent["split"] = split

    all_sentences.extend(sentences)

# Create DataFrame
df = pd.DataFrame(all_sentences)
print(f"Total sentences: {len(df)}")
print("Offensive distribution:")
print(df["offensive"].value_counts())

# Save to CSV
df.to_csv("data/interim/narabizi_treebank_raw.csv", index=False)
print("Saved to data/interim/narabizi_treebank_raw.csv")
