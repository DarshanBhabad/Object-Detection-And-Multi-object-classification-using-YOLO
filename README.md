# 🎯 SmartCity: Vehicle Detection & Analysis using YOLO26

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sstnUl5kfNvfffP1oZJxbSqoEs4_r-xm#scrollTo=SdTTMlvGyX-y)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Ultralytics YOLO26](https://img.shields.io/badge/Model-YOLO26-red)](https://docs.ultralytics.com/models/yolo26/)

## 👤 Author
**Darshan Ganesh Bhabad** *Deep Learning & Computer Vision Student*

---

## 🔗 Quick Links
* **Interactive Notebook**: [Run on Google Colab](https://colab.research.google.com/drive/1sstnUl5kfNvfffP1oZJxbSqoEs4_r-xm#scrollTo=SdTTMlvGyX-y)
* **Dataset Source**: [Vehicle Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/alkanerturan/vehicledetection)

---

## 📋 Project Overview
This project implements a real-time **Traffic & Road Safety Monitoring** system using the cutting-edge **YOLO26** architecture (released January 2026). The model is specifically trained to detect and categorize five vehicle classes: `Ambulance`, `Bus`, `Car`, `Motorcycle`, and `Truck`. 

By leveraging YOLO26's streamlined, NMS-free design, this project demonstrates a high-efficiency solution suitable for edge deployment in smart city infrastructures.

### Key Technical Features
* **End-to-End NMS-Free Inference**: YOLO26 eliminates the Non-Maximum Suppression post-processing step, resulting in faster and lighter deployment.
* **MuSGD Optimizer**: Utilizes the hybrid MuSGD optimizer (inspired by Kimi K2) for superior training stability and faster convergence.
* **DFL-Free Design**: Removed Distribution Focal Loss to broaden hardware compatibility for low-power edge devices.
* **Multi-Task Support**: Includes implementations for Object Detection, Instance Segmentation, and Image Classification.

---

## 🚀 Performance Metrics
The model was trained for **50 epochs** on a Google Colab Tesla T4 GPU environment.

| Task | Model | mAP@50 | mAP@50-95 | Inf. Time (ms) |
| :--- | :--- | :---: | :---: | :---: |
| **Detection** | `yolo26n` | **0.6335** | **0.4939** | 2.4ms |
| **Segmentation**| `yolo26n-seg`| 0.5330 | 0.3390 | 2.7ms |
| **Classification**| `yolo26n-cls`| N/A | N/A | 1.1ms |

---

## 🖼️ Results Visualization

### Training Progress
The training and validation curves indicate steady convergence of Box, Class, and DFL losses over the 50-epoch duration, successfully capturing the features of various vehicle types.


---

## 🌍 Application Domain: Traffic Monitoring
* **Problem**: High urban congestion and the need for prioritized emergency vehicle (Ambulance) detection.
* **Solution**: Automated classification of vehicle types to optimize traffic signal timings and emergency lanes.
* **Target Hardware**: Optimized for CPU-based edge devices due to YOLO26's 43% faster CPU inference speed compared to previous generations.

---

## 🛠️ How to Run
1.  **Install Ultralytics (Latest)**:
    ```bash
    pip install -U ultralytics
    ```
2.  **Load the Model**:
    ```python
    from ultralytics import YOLO
    model = YOLO('yolo26n.pt')
    ```
3.  **Run Inference**:
    ```python
    results = model.predict(source='your_image.jpg', conf=0.3)
    ```

---

## 📜 Acknowledgements
* **Alkanerturan** for the [Vehicle Detection Dataset](https://www.kaggle.com/datasets/alkanerturan/vehicledetection).
* **Ultralytics** for the YOLO26 framework.
* **Google Colab** for providing GPU compute resources.

---
