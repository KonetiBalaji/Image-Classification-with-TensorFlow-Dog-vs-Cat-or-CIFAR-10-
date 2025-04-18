# CIFAR-10 Image Classification with TensorFlow 

A deep learning project that classifies images from the CIFAR-10 dataset using a modern CNN architecture powered by `SeparableConv2D`, data augmentation, advanced callbacks, and efficient `tf.data` pipelines.

---

##  Overview

This project demonstrates:

- `tf.data` pipeline with `map()`, `shuffle()`, `prefetch()` for fast input
- Data augmentation using both `tf.image` and Keras `Random*` layers
- Efficient CNN using `SeparableConv2D` and `GlobalAveragePooling2D`
- `AdamW` optimizer + `CosineDecay` learning rate schedule
- `EarlyStopping`, `ModelCheckpoint`, and `TensorBoard` integration
- Saved model in `.keras` format
- Optionally export to `.tflite` for mobile/edge devices
- Optionally generate Markdown training report

---

##  Project Structure

```
.
├── image_classification_cifar10.py   # Main script
├── best_model.keras                  # Best model saved in modern Keras format
├── model_cifar10.tflite              # (Optional) TFLite model for edge deployment
├── training_report.md                # (Optional) Auto-generated training summary
├── logs/                             # TensorBoard logs
├── LICENSE                           # MIT license
├── requirements.txt                  # All dependencies
└── README.md                         # Project overview

---

##  Model Architecture

```text
Input Image →
  [Data Augmentation] →
  [SeparableConv2D → BN → SeparableConv2D → BN → MaxPool → Dropout] →
  [SeparableConv2D → BN → SeparableConv2D → BN → MaxPool → Dropout] →
  [GlobalAveragePooling2D → Dense(128) → Dropout → Dense(10)]
```

---

## Training Result Summary

| Metric           | Value       |
|------------------|-------------|
| Test Accuracy    | ~73.9%      |
| Test Loss        | ~0.75       |
| Epochs           | 50 (EarlyStopping) |
| Augmentation     | Flip, Rotate, Zoom, Contrast, Brightness |
| Optimizer        | AdamW       |
| LR Scheduler     | CosineDecay |
| Mixed Precision  | Optional    |

---

##  Setup Instructions

```bash
git clone https://github.com/your-username/cifar10-classifier.git
cd cifar10-classifier

# (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the training
python image_classification_cifar10.py
```

---

##  Run TensorBoard

```bash
tensorboard --logdir=logs
```

---

##  Optional Exports

- Save model in `.keras` format:
  ```python
  ModelCheckpoint("best_model.keras", save_best_only=True)
  ```

- Convert to `.tflite`:
  ```python
  converter = tf.lite.TFLiteConverter.from_saved_model("best_model.keras")
  tflite_model = converter.convert()
  with open("model_cifar10.tflite", "wb") as f: f.write(tflite_model)
  ```

---

##  Option D: Transfer Learning (Starter)

Uncomment to try:

```python
# base_model = tf.keras.applications.MobileNetV2(input_shape=(32, 32, 3),
#                                                include_top=False,
#                                                weights='imagenet')
# base_model.trainable = False
# model = Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     Dense(128, activation='relu'),
#     Dropout(0.3),
#     Dense(10, activation='softmax')
# ])
```

---

##  LICENSE

MIT License © 2025 Balaji Koneti

---

##  Acknowledgments

- [TensorFlow Docs](https://www.tensorflow.org/)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Keras Callbacks](https://keras.io/api/callbacks/)
```

# CIFAR-10 Training Report

**Model:** Custom CNN with SeparableConv2D and GlobalAveragePooling  
**Epochs:** 50  
**Optimizer:** AdamW  
**Learning Rate Decay:** CosineDecay  
**Callbacks Used:** EarlyStopping, ModelCheckpoint, TensorBoard  
**Final Test Accuracy:** 73.9%  
**Final Test Loss:** 0.755  
**Confusion Matrix and Misclassified Samples:** Visualized in training logs.  
**Model Format:** Saved in .keras (modern format), convertible to .tflite.