# 🌾 TinyML-Based Offline Smart Farming Assistant  
### Team 14 – Hackslayers

A **low-cost**, **portable**, and **offline-capable** embedded system designed for real-time **soil health analysis** and **pest detection**, tailored for **smallholder farmers in India**. Powered by **TinyML**, it enables intelligent, on-field agricultural decision-making without internet dependency.

---

## 🚜 Problem Statement

Smallholder farmers in rural India encounter persistent, high-impact challenges that hinder sustainable crop production and soil management:

- Non-scientific **fertilizer application** methods, leading to micronutrient depletion or over-enrichment.
- **Delayed detection** of pest infestations, resulting in reactive pesticide usage and reduced crop yield.
- **Manual irrigation patterns** not optimized to current soil conditions or crop water requirements.
- Existing **smart agriculture tools are internet-reliant and prohibitively expensive**.
- Lack of **personalized, multilingual** guidance suitable for low-tech rural environments.

### 💥 Impact
These systemic inefficiencies call for a **standalone**, **field-ready**, and **intelligent assistant** to empower farmers with data-driven insights — **without requiring mobile apps, cloud connectivity, or technical expertise**.

---

## 💡 Solution Overview

We designed a **TinyML-enabled handheld device** that integrates multiple environmental sensors and ML inference capabilities on microcontrollers. The system operates fully **offline**, performing **real-time data acquisition, normalization, prediction, and user feedback** — all on-device.

---

## 🔧 System Features

### 🧪 Soil Health Monitoring
**Sensors integrated** into the device collect real-time environmental data:
- **Soil pH Sensor (Analog/Probe-based)**: For acidity/alkalinity levels.
- **Soil Moisture Sensor (Capacitive/Resistive)**: Determines volumetric water content.
- **Soil Temperature Sensor (DS18B20 / DHT22)**: Monitors temperature stress on roots.
- **Electrical Conductivity (EC) Sensor**: Gauges nutrient availability/salinity.

> Sensor data is pre-processed (scaled/normalized) and passed to an **on-device classifier** (e.g., Random Forest or Decision Tree deployed using **TFLite Micro**) that provides actionable recommendations such as:
> - Fertilizer usage (type, quantity)
> - Irrigation frequency
> - Soil amendment advice

### 🦟 Pest & Disease Detection

#### 📷 Image-Based Classification
- **Hardware**: OV2640 camera module (2MP) connected to ESP32-S3.
- **Model**: Lightweight CNN (MobileNet/Custom) trained on pest/disease datasets (PlantVillage, IPM images).
- **Inference**: Real-time on-device classification to identify common pests (e.g., aphids, mites) or symptoms (e.g., blight, mildew).

#### 🎙️ Audio-Based Detection
- **Hardware**: INMP441 MEMS microphone (I2S) for field sound capture.
- **Signal Processing**:
  - Convert audio into **Mel-spectrograms** using embedded DSP pipeline.
- **Model**: SNN/CNN-based classifier to detect sounds of pest swarms or infestations.
- **Edge Inference**: Executed on-device using TFLite Micro or Edge Impulse-generated firmware.

### 📶 Offline Decision Support System

- Entire inference pipeline is deployed **on microcontroller hardware (ESP32-S3 / STM32F4 series)**.
- No dependency on:
  - Smartphones
  - Internet/cloud
  - Remote servers
- Benefits:
  - Ultra-low latency (under 100ms inference)
  - Fully secure and private
  - Energy-efficient

### 📲 Display & User Interface
- **OLED/I2C LCD (1.3")** screen displays:
  - Sensor values
  - Inference results
  - Suggested actions (fertilizer, pest treatment, water needs)
- **Multilingual interface support** in roadmap (Tamil, Hindi, etc.)
- **Future Expansion**:
  - Offline **Text-to-Speech (TTS)** using onboard DAC + mini speaker
  - Haptic feedback (vibration motor) for alerts

### 🔋 Power & Enclosure
- **Power Supply**:
  - Rechargeable **Li-ion Battery (3.7V 1200mAh)**
  - Optional **solar panel input (5V 500mA)**
- **Power Management** via onboard buck converter (AMS1117 / TP4056)
- **Form Factor**:
  - Enclosed in 3D-printed or laser-cut durable casing
  - Compact, weather-resistant, field-optimized

---

---

## 🏗️ Architecture Overview

The system is architected as an **embedded offline edge intelligence platform** combining sensor fusion, TinyML inference, and human-readable feedback — all inside a portable form factor.

### 🔧 Hardware Stack

| Component           | Role                                                                 |
|---------------------|----------------------------------------------------------------------|
| **ESP32-S3 / STM32** | Microcontroller Unit (MCU); handles inference, sensor I/O, display   |
| **Soil Sensor Kit** | Reads pH, moisture, temperature, EC via analog/digital interfaces     |
| **OV2640 Camera**   | Captures crop images for pest/disease classification                 |
| **INMP441 MEMS Mic**| Records audio for pest activity detection using audio ML             |
| **OLED 1.3” Display**| Displays actionable insights and alerts in real time                 |
| **Li-Ion Battery**  | Powers the device; rechargeable via USB or optional solar module     |
| **SD Card (optional)** | Local data logging, offline analysis storage                       |

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
