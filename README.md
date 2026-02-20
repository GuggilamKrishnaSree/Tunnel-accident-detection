# Real-Time Tunnel Accident Anticipation and Detection System

This project presents a real-time video-based system for **traffic accident anticipation and detection**, specifically designed for **tunnel-like environments** using deep spatio-temporal learning.

The system not only detects accidents after they occur, but also **anticipates potential accidents several frames in advance**, enabling early warnings and proactive safety measures.

---

## 🚀 Key Objectives
- Accident anticipation from video streams
- Real-time accident detection
- Tunnel environment simulation
- Explainable AI for safety-critical decisions
- Deployment-ready architecture for live inference

---

## 🧠 Core Features
- Video-based accident analysis
- Tunnel lighting and visibility simulation
- CNN + LSTM spatio-temporal architecture
- Sliding window–based accident anticipation
- Real-time alert generation
- Grad-CAM–based explainability
- Web-based and server-side deployment support

---

## 🏗️ System Architecture
1. Video input (CCTV / dashcam / live stream)
2. Tunnel environment simulation (training-time)
3. Spatial feature extraction using CNN (MobileNet)
4. Temporal modeling using LSTM
5. Accident anticipation and detection
6. Visual explanation using Grad-CAM
7. Alert generation and deployment

---

## 📊 Dataset
- **CarCrashDataset (CCD)** for accident videos
- Normal driving videos for non-accident scenarios
- Tunnel-like conditions simulated using image-level transformations
- Frame-level and sequence-level labeling strategy

---

## ⚙️ Technology Stack
- Python
- TensorFlow / PyTorch
- OpenCV
- FastAPI / Flask
- NumPy, Matplotlib
- VS Code

---

## 📌 Project Status
Currently under active development with planned enhancements including:
- Real-time video streaming interface
- Alert notification system
- Deployment on edge/cloud servers
