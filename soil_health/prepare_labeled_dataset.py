import pandas as pd

df = pd.read_csv("Plant_Parameters.csv")

def learn_thresholds(df):
    thresholds = {}
    for crop, group in df.groupby("Plant Type"):
        stats = {}
        for col in group.columns:
            if col != "Plant Type":
                mean = group[col].mean()
                std = group[col].std()
                stats[col] = (mean - std, mean + std)
        thresholds[crop] = stats
    return thresholds

thresholds = learn_thresholds(df)

def label_row(row):
    crop = row["Plant Type"]
    for feature, (low, high) in thresholds[crop].items():
        value = row[feature]
        if value < low or value > high:
            return 0
    return 1

df["Healthy"] = df.apply(label_row, axis=1)
df.to_csv("Labeled_Soil_Data.csv", index=False)
print("✅ Saved as Labeled_Soil_Data.csv")
