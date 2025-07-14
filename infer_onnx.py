# infer_onnx.py

import onnxruntime as ort
import numpy as np

# Sample input: Replace with real sensor values or test samples
input_data = np.array([[6.2, 0.3, 13, 120, 50, 25, 23, 34, 55]], dtype=np.float32)

# Load ONNX model
session = ort.InferenceSession("soil_model_20250710_150113.onnx")
 # replace with your actual file name
outputs = session.run(None, {"input": input_data})

# Get prediction
predicted_class = np.argmax(outputs[0], axis=1)[0]
print("🌱 Predicted Class Index:", predicted_class)
