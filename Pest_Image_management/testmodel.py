import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# === Config ===
MODEL_PATH = "/Users/devilphoenix/python/team_14/Pest_Image_management/ccmt_squeezenet_cleaned_20250719_102240_25epochs.pth"  # update!
CLASSES_PATH = "/Users/devilphoenix/python/team_14/Pest_Image_management/pest_classes.txt"
TEST_IMAGE = "/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/CCMT_Cleaned/healthy/Cashew_healthy_7.jpg"

# === Load class names ===
with open(CLASSES_PATH) as f:
    classes = [line.strip() for line in f.readlines()]

print(f"✅ Classes: {classes}")

# === Load model ===
model = models.squeezenet1_1(weights=None)
model.classifier[1] = nn.Conv2d(512, len(classes), kernel_size=(1,1))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load image ===
image = Image.open(TEST_IMAGE).convert("RGB")
x = transform(image).unsqueeze(0)

# === Predict ===
with torch.no_grad():
    output = model(x)
    probs = torch.nn.functional.softmax(output, dim=1)[0].numpy()

print(f"✅ Raw Probabilities: {probs}")
print(f"✅ Predicted Class: {classes[probs.argmax()]}")
print(f"✅ Confidence: {probs.max() * 100:.2f}%")
