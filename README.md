# 🌾 TinyML-Based Offline Smart Farming Assistant

A low-cost, portable, offline, intelligent farming companion for smallholder farmers in India, powered by TinyML and real-time sensor data.

---

## 🚜 Problem Statement

Smallholder farmers in rural India face daily challenges that threaten both crop yield and soil sustainability:

- Guesswork-based **fertilizer usage**, leading to nutrient imbalance.
- **Delayed pest detection**, resulting in heavy pesticide reliance and yield loss.
- **Habitual irrigation schedules**, causing overuse or water scarcity.
- **High-cost, internet-dependent smart farming tools** out of reach for most.
- **Lack of localized support** or personalized advisory systems in native languages.

These issues call for a simple, offline, field-ready, and intelligent system to support informed agricultural decisions at the grassroots level.

---

## 💡 Proposed Solution

We developed a **TinyML-powered handheld device** that provides offline, real-time insights into soil conditions and pest activity using sensor and vision/audio inputs. The device delivers actionable feedback directly to farmers in the field without needing internet access.

---

## 🔧 Features

### 🧪 Soil Health Monitoring
- Real-time readings for:
  - **Soil pH**
  - **Moisture level**
  - **Temperature**
  - **Electrical Conductivity (EC)**
- Data normalized and fed into on-device ML models to suggest corrective actions.

### 🦟 Pest & Disease Detection
- Image-based pest/disease classification using:
  - **Camera module (OV2640) + CNN model**
- Audio-based pest activity detection using:
  - **MEMS mic (INMP441) + spectrogram-based classifier**

### 📶 Offline Intelligent Decision Support
- All inference handled locally using:
  - **TinyML models (Edge Impulse / TFLite Micro)**
  - **MCU (ESP32-S3 or STM32)**
- No dependency on cloud or mobile apps for primary use.

### 📲 Display & Feedback
- Visual recommendations shown on:
  - **1.3" OLED/I2C LCD screen**
- Future scope: Local language **Text-to-Speech (TTS)** feedback via onboard speaker.

### 🔋 Energy Efficient & Portable
- Powered by:
  - **Rechargeable Li-ion battery**
  - Optional **solar charging support**
- Rugged, pocket-sized enclosure for field durability.

---
## 🤝 Team

- **Team 14: Hackslayers**


**Institution:** Shiv Nadar University, Chennai

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---


