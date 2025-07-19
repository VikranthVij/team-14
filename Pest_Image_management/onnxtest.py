import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

# === CONFIG ===
ONNX_PATH = "YOUR_SAVED_MODEL_25epochs.onnx"
IMG_PATH = "/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/CCMT_Cleaned/cashew_anthracnose/anthracnose175_.jpg"

# === Preprocess ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

img = Image.open(IMG_PATH).convert("RGB")
input_tensor = transform(img).unsqueeze(0).numpy()  # shape: [1, 3, 224, 224]

# === Load ONNX ===
session = ort.InferenceSession(ONNX_PATH)

# === Get input & output names ===
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# === Run ===
outputs = session.run([output_name], {input_name: input_tensor})
pred = np.argmax(outputs[0], axis=1)

print(f"✅ ONNX prediction: {pred[0]}")
