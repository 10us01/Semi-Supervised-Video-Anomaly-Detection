# Semi-Supervised Video Anomaly Detection (Pixel Play 2026)

![AUC Score](https://img.shields.io/badge/Public_Leaderboard_AUC-0.79-brightgreen?style=for-the-badge&logo=kaggle)

**Hosted by:** Vision and Language Group (VLG), IIT Roorkee  
**Status:** Completed (January 2026)

## 📌 Project Overview
This repository contains the official implementation for the **Pixel Play 2026** anomaly detection challenge. The objective is to develop a robust system capable of identifying abnormal events in video surveillance footage—such as unauthorized entries, sudden running, or falls—using a semi-supervised learning framework.

The core challenge of Video Anomaly Detection (VAD) is the lack of anomalous training data. Our system defines "normality" based exclusively on a training set of standard activities and flags any significant deviations during inference as potential anomalies.

## 🚀 Key Features
* **Two-Stream Architecture:** Simultaneously processes the entire scene (Global) and individual actors (Local) to detect both environmental and behavioral anomalies.
* **Foundation Model Backbone:** Leverages **CLIP (ResNet-101)** to extract rich, semantic features that capture high-level concepts better than traditional ImageNet-trained models.
* **Human-Centric Analysis:** Integrates **YOLOv5s** to isolate and analyze human poses independently from background noise.
* **Memory Efficiency:** Implements **Greedy Coreset Subsampling** to retain only the most diverse 1% of feature vectors, optimizing memory usage without sacrificing accuracy.
* **Robust Scoring:** Uses a 3-seed ensemble strategy combined with **Gaussian Temporal Smoothing** to ensure stable and noise-free anomaly detection scores.

## 🏗️ Technical Architecture
The system employs a unified **VideoPatchCore** framework that forks into two parallel processing streams:

### 1. The Global Stream (Context)
* **Input:** Full $224 \times 224$ video frames.
* **Function:** Models the background environment.
* **Use Case:** Detects scene-level irregularities, such as a vehicle entering a pedestrian walkway.

### 2. The Local Stream (Behavior)
* **Input:** Dynamic crops of human subjects detected by **YOLOv5s**.
* **Function:** Models specific human actions and poses.
* **Use Case:** Detects behavioral anomalies like fighting, falling, or loitering.

**Feature Extraction:** Features are extracted from **Layer 2** and **Layer 3** of the CLIP ResNet-101 backbone. These features are fused and globally pooled into compact embedding vectors for efficient comparison.

## 📂 Repository Structure
```text
├── Final_codes/training_notebook/   # Main pipeline (final submission - Feature Extraction + Memory Bank creation)
├── Experiments/                     # Most significant experiments with different models
├── requirements.txt                 # List of Python dependencies
└── README.md                        # Project Documentation
