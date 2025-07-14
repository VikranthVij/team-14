
import onnxruntime as ort
import numpy as np

# Load session
session = ort.InferenceSession("YOUR_SAVED_MODEL.onnx")

# Print input details
print("Inputs:", session.get_inputs())
print("Outputs:", session.get_outputs())

# Create a fake input
dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": dummy})
print("✅ ONNX output:", outputs)
