import streamlit as st
from PIL import Image
import numpy as np
import os
import torch
from torchvision import transforms, models
import torch.nn as nn

# === Page config ===
st.set_page_config(
    page_title="Smart Farming Assistant",
    page_icon="🌾",
    layout="wide"
)

# === Custom CSS ===
css_path = os.path.join(os.path.dirname(__file__), "style.css")
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# === Recommendations ===
RECOMMENDATIONS = {
    "cashew_anthracnose": "Use copper fungicide sprays. Prune affected parts. Maintain canopy airflow.",
    "cashew_gumosis": "Improve drainage. Avoid injuries to bark. Apply fungicides to wounds.",
    "cashew_leaf_miner": "Spray neem oil or introduce parasitoids. Remove infested leaves.",
    "cashew_red_rust": "Apply Bordeaux mixture. Remove heavily infected shoots.",
    "cassava_bacterial_blight": "Use clean cuttings. Remove infected plants. Ensure good spacing.",
    "cassava_brown_spot": "Practice crop rotation. Use resistant varieties. Remove debris.",
    "cassava_green_mite": "Spray recommended acaricides if severe. Encourage natural predators.",
    "cassava_mosaic": "Use virus-free planting material. Remove infected plants immediately.",
    "healthy": "No infection detected. Keep monitoring and follow good agricultural practices.",
    "maize_fall_armyworm": "Use pheromone traps. Apply biopesticides like Bacillus thuringiensis.",
    "maize_grasshoper": "Handpick if low infestation. Use recommended insecticides if needed.",
    "maize_leaf_beetle": "Spray recommended insecticides early. Rotate crops.",
    "maize_leaf_blight": "Apply fungicides. Use resistant varieties. Avoid water stress.",
    "maize_leaf_spot": "Use certified seeds. Rotate crops. Apply fungicide if severe.",
    "maize_streak_virus": "Control vectors (leafhoppers). Use tolerant hybrids.",
    "tomato_leaf_blight": "Spray fungicides. Remove lower infected leaves. Improve spacing.",
    "tomato_leaf_curl": "Control whiteflies with yellow sticky traps and insecticides.",
    "tomato_septoria_leaf_spot": "Apply protectant fungicides. Use drip irrigation. Remove debris.",
    "tomato_verticulium_wilt": "Practice crop rotation. Remove infected plants. Solarize soil."
}

# === Load trained model ===
@st.cache_resource
def load_model():
    num_classes = 19
    model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
    model.num_classes = num_classes
    model.load_state_dict(torch.load(
        "/Users/devilphoenix/python/team_14/Pest_Image_management/ccmt_squeezenet_cleaned_20250719_102240_25epochs.pth",
        map_location="cpu"
    ))
    model.eval()
    return model

model = load_model()

# === Load classes ===
with open("/Users/devilphoenix/python/team_14/Pest_Image_management/pest_classes.txt") as f:
    CLASSES = [line.strip() for line in f]

# === Transform ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Main UI ===
st.title("Smart Farming Assistant 🌾")

with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["🏠 Home", "🪲 Pest Detection", "🌱 Soil Analysis", "ℹ️ About"])

# === Pages ===
if page == "🏠 Home":
    st.header("Welcome to Smart Farming Assistant")
    st.write("""
    <div style='text-align: center; font-size: 1.2em;'>
    🌾 A portable, offline-capable solution for real-time agricultural analysis 🌾
    </div>
    This application provides:
    - 🪲 Pest and disease detection in crops
    - 🌱 Soil health analysis
    - 🔄 Offline operation using TinyML models
    - 📱 Real-time feedback for farmers
    """, unsafe_allow_html=True)

elif page == "🪲 Pest Detection":
    st.header("Pest Detection 🪲")
    st.write("Upload an image of a crop leaf for pest detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        input_tensor = transform(image).unsqueeze(0)

        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
            predicted_class_idx = np.argmax(probs)
            confidence = probs[predicted_class_idx] * 100

            predicted_class_name = CLASSES[predicted_class_idx]
            recommendation = RECOMMENDATIONS.get(predicted_class_name, "No specific advice found.")
            pest_probability = 0.0 if predicted_class_name == "healthy" else confidence

        risk_label = "High Risk" if pest_probability > 50 else "Low Risk"
        conf_label = "Very Confident" if confidence > 70 else "Low Confidence"

        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pest Probability", f"{pest_probability:.2f}%", delta=risk_label)
        with col2:
            st.metric("Confidence", f"{confidence:.2f}%", delta=conf_label)

        st.info(f"""
        ✅ **Detected:** {predicted_class_name.replace("_", " ").title()}  
        💡 **Recommendation:** {recommendation}
        """)

elif page == "🌱 Soil Analysis":
    st.header("Soil Analysis 🌱")
    st.write("Enter soil sensor readings for analysis")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Soil Parameters")
        ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        moisture = st.slider("Moisture Level", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    with col2:
        st.subheader("Environmental")
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        ec = st.slider("Electrical Conductivity", min_value=0.0, value=2.0, step=0.1)

    if st.button("🔍 Analyze Soil", type="primary"):
        st.subheader("Soil Analysis Results 📊")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Soil Condition", value="Healthy", delta="Optimal")
        with col2:
            st.metric(label="Fertilizer Recommendation", value="NPK 19-19-19", delta="Balanced")

        st.success("""
        Soil Health Summary:
        - pH: Optimal range
        - Moisture: Adequate
        - Temperature: Ideal
        - EC: Suitable

        Recommendations:
        - Continue current fertilization practices
        - Maintain regular irrigation
        - Monitor soil regularly
        """)

elif page == "ℹ️ About":
    st.header("About")
    st.write("""
    <div style='text-align: center; font-size: 1.2em;'>
    🌾 Smart Farming Assistant Project 🌾
    </div>

    This application is designed to help smallholder farmers in India with real-time agricultural insights.

    Features:
    - 🪲 Pest detection using CNN models
    - 🌱 Soil health analysis
    - ⚡ Offline operation using TinyML
    - 📱 Real-time feedback

    Built by Hackslayers Team:
    - Vikranth V
    - Varsha Pillai M
    - Nittin Balajee S
    - Lithikha B
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 20px;'>
    <a href='https://github.com/annam-ai-iitropar/team_14' target='_blank'>
    <button style='background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer;'>
    View on GitHub
    </button>
    </a>
    </div>
    """, unsafe_allow_html=True)
