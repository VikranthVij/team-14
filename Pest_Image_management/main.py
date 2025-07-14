import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError, ImageFile

# Allow PIL to load broken images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === CONFIG ===
DATA_DIR = "/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset"
BATCH_SIZE = 32
EPOCHS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_NAME = f"ccmt_squeezenet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === SAFE ImageFolder ===
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError):
            print(f"❌ Corrupted image skipped: {path}")
            return self.__getitem__((index + 1) % len(self.samples))
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

dataset = SafeImageFolder(DATA_DIR, transform=transform)
num_classes = len(dataset.classes)
print(f"✅ Found {num_classes} classes: {dataset.classes}")

# Save class names
with open("pest_classes.txt", "w") as f:
    for cls in dataset.classes:
        f.write(f"{cls}\n")

# === TRAIN/VAL SPLIT ===
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# === MODEL ===
model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
model.num_classes = num_classes
model = model.to(DEVICE)

# === LOSS & OPTIMIZER ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# === TRAINING ===
train_losses, val_losses, val_accuracies = [], [], []

print(f"🔍 Training on device: {DEVICE} (SqueezeNet 1.1)")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_losses.append(val_loss / len(val_loader))
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2f}%")

# === SAVE MODEL ===
torch.save(model.state_dict(), MODEL_SAVE_NAME)
print(f"✅ Model saved as {MODEL_SAVE_NAME}")

# === PLOT ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val Accuracy")
plt.title("Validation Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig("ccmt_squeezenet_training_plot.png")
plt.show()
