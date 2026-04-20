# ❄️ Snow Pole Detection using YOLOv8

## 📌 Overview

This project focuses on detecting **snow poles** in road images using **YOLOv8 object detection models**.
Snow poles are **thin, small, and high-aspect-ratio objects**, making them a challenging detection problem.

The project includes:

* Exploratory Data Analysis (EDA)
* Training multiple YOLOv8 models
* Model evaluation and comparison
* Test set qualitative analysis

---

## 📂 Project Structure

```
snow_pole_project/
├── experiments/              # EDA outputs (plots, analysis)
├── runs/                    # YOLO training & evaluation outputs
├── results/                 # Clean results (recommended)
├── eda.ipynb                # Main EDA notebook
├── requirements.txt         # Dependencies
├── README.md
```

---

## 📂 Dataset

The dataset consists of road images containing **snow poles**, which serve as visual markers in snowy environments.

### 📊 Dataset Size

* ~322 training images
* ~92 validation images
* Test set (unlabeled)



### 🗂️ Structure

```
dataset/
├── train/
│   ├── images/
│   ├── labels/
├── valid/
│   ├── images/
│   ├── labels/
├── test/
    ├── images/
```

* **Train / Validation**: images with YOLO-format annotations
* **Test**: images only (no labels)

### ⚠️ Notes

* The **test set does not contain labels**, therefore:

  * Validation set → used for quantitative evaluation (mAP)
  * Test set → used for qualitative analysis

* The dataset is **not included in this repository** because it was provided by our course instructor and is subject to usage restrictions.

---

## 🔍 Exploratory Data Analysis (EDA)

Key findings:

* Most images contain **1–2 poles**
* Objects are **very small in width**
* High **height-to-width ratio**
* Strong positional patterns in images

These characteristics directly impact model performance and localization accuracy.

---

## 🤖 Models Trained

| Model   | Image Size | Epochs |
| ------- | ---------- | ------ |
| YOLOv8s | 640        | 50     |
| YOLOv8m | 1024       | 100    |
| YOLOv8m | 1280       | 100    |

---

## 📈 Results (Validation Set)

| Model          | Precision | Recall    | mAP50     | mAP50-95  |
| -------------- | --------- | --------- | --------- | --------- |
| YOLOv8s (640)  | 0.891     | 0.779     | 0.829     | 0.494     |
| YOLOv8m (1024) | **0.990** | 0.920     | **0.989** | **0.689** |
| YOLOv8m (1280) | 0.924     | **0.929** | 0.976     | 0.689     |

---

## 🏆 Best Model

👉 **YOLOv8m (1024)**

### Why?

* Highest mAP50
* Strong precision and recall
* Same mAP50-95 as 1280 but more efficient

---

## 🧠 Key Insights

* Increasing resolution (640 → 1024) significantly improves performance
* Further increase (1024 → 1280) shows **diminishing returns**
* Gap between mAP50 and mAP50-95 indicates:

  * Good detection
  * Hard precise localization

### Reason (from EDA):

Snow poles are **thin and small**, making bounding box precision difficult.

---

## 🧪 Test Set Evaluation

Since test labels are unavailable:

* Only **predictions** were generated
* Used for **qualitative analysis**

### Observations:

* Model generalizes well to unseen data
* Detects most poles accurately
* Struggles with:

  * very small / distant poles
  * exact bounding box alignment

---

## 🚀 How to Run

### Install dependencies

```
pip install -r requirements.txt
```

---

### Train model

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
yolo detect val \
  model=path/to/best.pt \
  data=dataset/data.yaml \
  split=val
```

---

### Predict on test set

```
yolo detect predict \
  model=path/to/best.pt \
  source=dataset/test/images
```

---

## 📌 Future Improvements

* Enable data augmentation
* Increase dataset size
* Improve small object detection
* Try anchor tuning or advanced augmentations

---

## 👩‍💻 Author

Payal Singla
