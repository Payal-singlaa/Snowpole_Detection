# Snow Pole Detection using YOLOv8

## 📌 Overview
This project focuses on detecting snow poles from LiDAR-based images using deep learning models.

Snow poles are critical for autonomous driving in snowy environments, as they indicate road boundaries when lane markings are not visible.

---

## 🚀 Results

| Model        | Image Size | Epochs | mAP50 |
|-------------|----------|--------|------|
| YOLOv8s     | 1024     | 50     | ~0.88 |
| YOLOv8m     | 1024     | 80     | (running...) |

---

## 🧠 Approach

- Used YOLOv8 for object detection
- Trained on LiDAR-based dataset
- Focused on improving detection of small objects

---

## ⚙️ How to Run

```bash
yolo detect train model=yolov8s.pt data=data.yaml epochs=50 imgsz=1024