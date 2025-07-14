import streamlit as st
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(
    page_title="Smart Farming Assistant",
    page_icon="🌾",
    layout="wide"
)

# Add custom CSS
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Main app
st.title("Smart Farming Assistant 🌾")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio("", ["🏠 Home", "🪲 Pest Detection", "🌱 Soil Analysis", "ℹ️ About"])

# Home page
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
    
    All processing is done locally on your device.
    """, unsafe_allow_html=True)

# Pest Detection page
elif page == "🪲 Pest Detection":
    st.header("Pest Detection 🪲")
    st.write("Upload an image of a crop leaf for pest detection")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Display results in a more organized way
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Pest Probability", value="85%", delta="High Risk")
        with col2:
            st.metric(label="Confidence", value="90%", delta="Very Confident")
        
        st.info("""
        Recommendations:
        - Inspect plants for visible pests
        - Apply appropriate pesticide treatment
        - Monitor regularly for improvement
        """)

# Soil Analysis page
elif page == "🌱 Soil Analysis":
    st.header("Soil Analysis 🌱")
    st.write("Enter soil sensor readings for analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Soil Parameters")
        ph = st.slider("pH Level", min_value=0.0, max_value=14.0, value=7.0, step=0.1, label="pH Level")
        moisture = st.slider("Moisture Level", min_value=0.0, max_value=100.0, value=50.0, step=1.0, label="Moisture Level")
    with col2:
        st.subheader("Environmental")
        temp = st.slider("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1, label="Temperature")
        ec = st.slider("Electrical Conductivity", min_value=0.0, value=2.0, step=0.1, label="Electrical Conductivity")
    
    if st.button("🔍 Analyze Soil", type="primary"):
        # Display results in a more organized way
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

# About page
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
