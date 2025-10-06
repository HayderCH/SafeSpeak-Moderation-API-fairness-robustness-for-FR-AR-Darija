import pandas as pd

# Load train and test CSVs
train = pd.read_csv(
    "data/raw/public/Arabic_hatespeech-master/" "Arabic_hatespeech-master/train.csv"
)
test = pd.read_csv(
    "data/raw/public/Arabic_hatespeech-master/" "Arabic_hatespeech-master/test.csv"
)

# Collect all IDs
ids = list(train["id"]) + list(test["id"])
print(f"Total IDs: {len(ids)}")

# Write to file
with open("data/interim/albadi_tweet_ids.txt", "w") as f:
    f.write("\n".join(map(str, ids)))

print("IDs written to data/interim/albadi_tweet_ids.txt")
