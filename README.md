# 🌾 TinyML-Based Offline Smart Farming Assistant
### Team 14 – Hackslayers

## Project Overview
A **low-cost**, **portable**, and **offline-capable** embedded system designed for real-time **soil health analysis** and **pest detection**, tailored for **smallholder farmers in India**. Powered by **TinyML**, it enables intelligent, on-field agricultural decision-making without internet dependency.

## 📋 Project Structure
```
smart_farming_assistant_annam/
├── /src/           # Source code
├── /app/           # Application code
├── /tests/         # Test suites
├── /docs/          # Documentation
│   ├── /cards/    # Project and model cards
│   └── /notebooks/ # Notebooks and analysis
├── /data/          # Data and datasets
├── /notebooks/     # Jupyter notebooks
└── /logs/          # Application logs
```

## 🛠️ Setup Instructions

### Prerequisites
- Python 3.8+
- ESP-IDF for ESP32 development
- Arduino IDE or PlatformIO
- TensorFlow Lite for Microcontrollers

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/smart_farming_assistant_annam.git
cd smart_farming_assistant_annam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
```bash
bash data/download.sh
```

4. Build the firmware:
```bash
idf.py build
```

## 📊 Performance Metrics
Detailed metrics can be found in [/docs/cards/ml-metrics.json](cci:7://file:///c:/Users/lithi/Desktop/ropar/team_14/docs/cards/ml-metrics.json:0:0-0:0)

## 📄 Documentation
- Project details: [/docs/cards/project-card.ipynb](cci:7://file:///c:/Users/lithi/Desktop/ropar/team_14/docs/cards/project-card.ipynb:0:0-0:0)
- Model metrics: [/docs/cards/ml-metrics.json](cci:7://file:///c:/Users/lithi/Desktop/ropar/team_14/docs/cards/ml-metrics.json:0:0-0:0)
- Architecture: [/docs/architecture.png](cci:7://file:///c:/Users/lithi/Desktop/ropar/team_14/docs/architecture.png:0:0-0:0)
- Notebooks: [/notebooks/](cci:7://file:///c:/Users/lithi/Desktop/ropar/team_14/notebooks:0:0-0:0)

## 📖 License
This project is licensed under the MIT License - see the LICENSE file for details.

### 🧠 Software & ML Stack

| Layer              | Tool / Framework                         | Purpose                                       |
|-------------------|-------------------------------------------|-----------------------------------------------|
| Data Collection    | Arduino IDE, Python, Edge Impulse         | Collect labeled datasets (images, audio, soil)|
| Model Training     | Edge Impulse Studio / TensorFlow Lite     | Train models (CNN, Spectrogram classifiers)   |
| On-Device Inference| TensorFlow Lite for Microcontrollers (TFLM)| Run optimized ML models on MCU                |
| Firmware           | C/C++ + PlatformIO / Arduino Framework    | Sensor drivers, pre/post-processing, control  |
| UI/UX              | OLED / LCD via I2C                        | Feedback interface (English, Local Lang. TTS*)|

\* *TTS in Tamil/Hindi via on-board DAC and speaker is part of future scope.*

---

## 🔍 Core Functional Modules

### 🧪 Soil Health Monitoring

- **Sensors Used:** Analog soil pH, Capacitive moisture sensor, DS18B20 (temperature), EC probe.
- **Processing:** All sensor values normalized and passed through a decision tree or linear regression TinyML model to classify soil condition (e.g., acidic, saline, dry).
- **Output:** Displays optimal fertilizer suggestions and irrigation advice.

### 🦟 Pest & Disease Detection

#### 🔹 Image-Based (CNN)
- **Input:** Crop leaf image from OV2640 camera.
- **Model:** Optimized Convolutional Neural Network (Edge Impulse).
- **Classes:** Common pest/disease categories like blight, aphids, mildew, etc.
- **Preprocessing:** Grayscale, 96x96 downscaling.

#### 🔹 Audio-Based (Spectrogram Classifier)
- **Input:** Audio captured from MEMS mic (INMP441).
- **Processing:** FFT-based spectrogram generated onboard using CMSIS-DSP or custom logic.
- **Inference:** Classify presence of pests (e.g., chirping insects, borers) or anomalies.
- **Output:** Alert user if pest activity is detected.

### 📶 Offline Intelligence & Decision Support

- All inference runs on-device using **TFLite Micro + quantized models**.
- No external cloud connection required.
- Memory and computation optimized using **CMSIS-NN / TFLM** interpreter with fixed-point arithmetic.
- Fail-safe defaults and fallback logic (e.g., for corrupt sensor readings or unclear model predictions).

---

## 📲 Human-Centered Feedback System

- **Visual Output:** 
  - 1.3” OLED displays:
    - Soil diagnosis & recommendations
    - Pest alerts (with image preview)
    - Icons and simple messages for low-literacy users
- **Planned Extension:**
  - **Audio Output:** Local language (Tamil/Hindi) TTS using DAC + mini speaker.
  - Audio feedback generated offline with pre-encoded phoneme playback or via uSpeech.

---

## 🔋 Power Management

- **Power Source:** 3.7V 18650 Li-ion Battery.
- **Efficiency Measures:**
  - DeepSleep modes to preserve battery during inactivity.
  - Low-power sensor polling cycles.
- **Charging:** Via USB or solar panel (6V/1W input supported).
- **Enclosure:** 3D-printed, dust-resistant ABS shell with waterproofing planned.

---

## 🚀 Future Roadmap

- ✅ Add multilingual TTS module with pre-recorded audio banks
- ✅ Develop Android interface for batch sync/data export (optional)
- 🔜 Extend pest dataset with local crops (paddy, millets, etc.)
- 🔜 Add GPS + crop calendar integration for seasonal alerts
- 🔜 Push OTA updates for firmware using LoRa/mesh hubs

---
## 🤝 Team

**Team 14: Hackslayers**  
Shiv Nadar University, Chennai  

Members
- Vikranth V
- Varsha Pillai M
- Nittin Balajee S
- Lithikha B

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for more details.
