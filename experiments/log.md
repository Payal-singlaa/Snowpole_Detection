## 🧪 Experiment Log

---

## Experiment 1: YOLOv8s (640)

**Configuration**

* Model: YOLOv8s
* Image Size: 640
* Epochs: 50

**Results**

* mAP50: ~0.83–0.84
* mAP50-95: ~0.46–0.49

**Observations**

* Stable and smooth convergence
* No noticeable overfitting
* Struggles with small and thin objects
* Limited by lower resolution and smaller model capacity

---

##  Experiment 2: YOLOv8m (1024)

**Configuration**

* Model: YOLOv8m
* Image Size: 1024
* Epochs: 100

**Results**

* mAP50: ~0.98–0.99
* mAP50-95: ~0.66–0.69

**Observations**

* Significant improvement over YOLOv8s
* Higher resolution improves detection of small objects
* Better localization performance
* Stable training with no overfitting
* Best overall trade-off between performance and efficiency

---

##  Experiment 3: YOLOv8m (1280)

**Configuration**

* Model: YOLOv8m
* Image Size: 1280
* Epochs: 100

**Results**

* mAP50: ~0.97–0.98
* mAP50-95: ~0.68–0.69

**Observations**

* Slight improvement in localization (mAP50-95)
* Minimal gain compared to 1024 resolution
* Higher computational cost
* Diminishing returns beyond 1024

---

## Final Comparison

| Model   | Image Size | mAP50     | mAP50-95  |
| ------- | ---------- | --------- | --------- |
| YOLOv8s | 640        | ~0.84     | ~0.48     |
| YOLOv8m | 1024       | **~0.99** | **~0.69** |
| YOLOv8m | 1280       | ~0.98     | ~0.69     |

---

##  Key Insights

* Increasing image resolution significantly improves performance
* 1024 resolution provides the best balance between accuracy and efficiency
* Further increase to 1280 yields only marginal gains
* Snow poles are **thin, high-aspect-ratio objects**, making precise localization challenging
* The gap between mAP50 and mAP50-95 highlights bounding box precision limitations

---

##  Conclusion

**YOLOv8m (1024)** is the best-performing model for this task.

It achieves:

* Highest mAP50
* Strong mAP50-95
* Efficient training compared to 1280

---
