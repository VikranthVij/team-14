import onnxruntime as ort
import numpy as np
import pandas as pd

df = pd.read_csv("Plant_Parameters.csv")
feature_columns = df.drop(columns=["Plant Type"]).columns.tolist()

def learn_thresholds(df):
    crop_thresholds = {}
    grouped = df.groupby("Plant Type")
    for crop, group in grouped:
        stats = {}
        for col in group.columns:
            if col != "Plant Type":
                mean = group[col].mean()
                std = group[col].std()
                stats[col] = (mean - std, mean + std)
        crop_thresholds[crop] = stats
    return crop_thresholds

thresholds = learn_thresholds(df)

def generate_recommendations(crop, input_features, thresholds, columns):
    messages = []
    for i, feature in enumerate(columns):
        value = input_features[i]
        low, high = thresholds[crop][feature]
        if value < low:
            messages.append(f"{feature}: ❌ Too Low — Suggest Increase")
        elif value > high:
            messages.append(f"{feature}: ❌ Too High — Suggest Decrease")
        else:
            messages.append(f"{feature}: ✅ OK")
    return messages

# ✨ Replace this with actual sensor reading
sensor_data = [6.2, 0.3, 13, 120, 50, 16, 23, 72, 55]  # dummy input
selected_crop = "Corn"
input_array = np.array([sensor_data], dtype=np.float32)

session = ort.InferenceSession("health_model_20250710_223125.onnx")
  # update name
result = session.run(None, {"input": input_array})
pred = np.argmax(result[0], axis=1)[0]

print(f"🌱 Crop Selected: {selected_crop}")
if pred == 1:
    print("✅ Soil is Healthy")
else:
    print("⚠️ Soil is Unhealthy — Recommendations:")
    recs = generate_recommendations(selected_crop, sensor_data, thresholds, feature_columns)
    for r in recs:
        print(" -", r)
