import torch
import torch.nn as nn
from torchvision import models

# === Your same config ===
num_classes = 19  # update if needed
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 1) Define same SqueezeNet ===
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
model.num_classes = num_classes
model = model.to(DEVICE)

# === 2) Load your saved weights ===
model.load_state_dict(torch.load("ccmt_squeezenet_cleaned_20250719_102240_25epochs.pth", map_location=DEVICE))
model.eval()

print("✅ Weights loaded!")

# === 3) Create dummy input ===
dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)

# === 4) Export to ONNX ===
onnx_file_name = "YOUR_SAVED_MODEL_25epochs.onnx"

torch.onnx.export(
    model,
    dummy_input,
    onnx_file_name,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
    export_params=True,
    opset_version=11
)

print(f"✅ Done! ONNX model saved as: {onnx_file_name}")
