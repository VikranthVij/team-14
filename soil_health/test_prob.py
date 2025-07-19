import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# === CONFIG ===
MODEL_PATH = "/Users/devilphoenix/python/team_14/Pest_Image_management/ccmt_squeezenet_aug_20250718_204931_30 epochs.pth"
IMAGE_PATH = "/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset/Maize/healthy/healthy7_.jpg"  # << REPLACE with your image

NUM_CLASSES = 4  # set this to your number of classes

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1))
model.num_classes = NUM_CLASSES

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

print(f"✅ Model loaded on {device}")

# === Image transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load image ===
image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# === Inference ===
with torch.no_grad():
    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
    predicted_class = int(probs.argmax())
    confidence = probs.max() * 100

print(f"✅ Raw Probabilities: {probs}")
print(f"✅ Predicted Class: {predicted_class}")
print(f"✅ Confidence: {confidence:.2f}%")
