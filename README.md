# ❄️ Snow Pole Detection using Faster R-CNN

## 📌 Overview

This project explores **snow pole detection** using a **Faster R-CNN** model.

Unlike YOLO (single-stage detector), Faster R-CNN is a **two-stage detector**, making it better suited for analyzing detection behavior, especially for **small and challenging objects**.

Snow poles are:
- thin  
- small  
- vertically elongated  

→ making this a **challenging object detection problem**

---

## 🎯 Objective

- Train a Faster R-CNN model on the dataset  
- Analyze performance in terms of **precision and recall**  
- Study behavior on **small object detection**  
- Compare with YOLO-based approach  

---

## 📂 Project Structure

```
snow_pole_project/
├── train.py # Training script
├── dataset.py # Custom dataset loader
├── model.py # Faster R-CNN model setup
├── eval_utils.py # Evaluation utilities
├── predict.py # Inference + visualization
├── results/rcnn/ # Prediction outputs
├── README.md
```


---

## 📂 Dataset

Same dataset as YOLO branch:

- ~322 training images  
- ~92 validation images  
- Test set (unlabeled)

### ⚠️ Important

- Labels are in **YOLO format**, converted for RCNN usage  
- Test set is used only for **qualitative evaluation**

---

## 🤖 Model

Model used:

👉 `Faster R-CNN with ResNet50-FPN backbone`

- Pretrained on COCO  
- Fine-tuned for **1 class (snow pole)**  

---

## ⚙️ Training Setup

- Optimizer: SGD  
- Learning rate: 0.005  
- Momentum: 0.9  
- Batch size: 4  
- Epochs: 25  
- Device: CUDA (GPU)

---

## 📈 Results

### Best Model (based on validation)

- **Precision:** ~0.80  
- **Recall:** ~0.75 (early training peak)

---

## 🧠 Key Observations

### 1. Precision–Recall Tradeoff

- Early training → high recall (detects more poles)  
- Later training → higher precision, lower recall  

👉 Model becomes more conservative over time

---

### 2. Small Object Challenge

- Many poles are extremely small  
- RCNN struggles to consistently detect all instances  

---

### 3. Overfitting Behavior

- Increasing epochs improves precision  
- But reduces recall → missing objects  

---

## 🔍 Comparison with YOLO

| Model | Precision | Recall | Speed | Behavior |
|------|----------|--------|------|---------|
| YOLOv8 | High | High | Fast ⚡ | Balanced |
| Faster R-CNN | Medium | High* | Slow 🐢 | Recall-focused |

\*High recall achieved in early epochs

---

## 🧪 Evaluation

- Custom evaluation implemented  
- Metrics:
  - Precision  
  - Recall  

- Validation set used for performance tracking  

---

## 🔎 Predictions

Predictions generated on test images:

- Good at detecting visible poles  
- Misses:
  - very small poles  
  - distant objects  

---

## 🚀 How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```
---

### Train

```
yolo detect train \
  data=dataset/data.yaml \
  model=yolov8m.pt \
  imgsz=1024 \
  epochs=100
```

---

### Validate model

```
python train.py
```

---

### Run Inference

```
python predict.py
```

---

## 📌 Key takeaways

* RCNN provides better interpretability of detection behavior
* Shows clear precision vs recall tradeoff
* Struggles with small object detection compared to YOLO

---

## 👩‍💻 Author

Payal Singla