import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms

session = ort.InferenceSession("YOUR_SAVED_MODEL_30epochs.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Use same transform as val set
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load real test image
img = Image.open("/Users/devilphoenix/Vikranth/Hackathons/Annam.ai/Datasets/Dataset for Crop Pest and Disease Detection/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm6_.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0).numpy()

outputs = session.run([output_name], {input_name: input_tensor})
pred = np.argmax(outputs[0], axis=1)
print(f"✅ ONNX predicted class: {pred[0]}")
