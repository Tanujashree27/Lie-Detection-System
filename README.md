
# Facial Expression-Based Lie Detection Using Deep Learning.

This project uses deep learning techniques to classify facial expressions from images into two categories: **Truth** and **Lie**. It leverages transfer learning with MobileNetV2 to achieve robust performance on a Kaggle dataset.

---

## 📁 Dataset

The dataset used is from [Kaggle](https://www.kaggle.com), consisting of facial expression images labeled as either:

* `truth`
* `lie`

### Dataset Summary:

* **Training Images**: 14,108
* **Validation Images**: 3,526.
* **Classes**: 2 (truth, lie)

---

## 🧠 Model Architecture

We use **MobileNetV2** as a feature extractor (with frozen weights), followed by custom classification layers:

```
Input Image (128x128x3)
↓
MobileNetV2 (pretrained, not trainable)
↓
GlobalAveragePooling2D
↓
BatchNormalization
↓
Dense (64 units, ReLU)
↓
Dropout (0.5)
↓
Dense (1 unit, Sigmoid)
↓
Output: Truth / Lie
```

---

## 📊 Training Configuration

* **Loss Function**: Binary Crossentropy
* **Optimizer**: Adam
* **Metrics**: Accuracy
* **Epochs**: 20
* **Batch Size**: 32
* **Image Size**: 128x128
* **Augmentation**: Horizontal flip, rotation, zoom

---

## 🧪 Training Output

The model achieved:

* Training accuracy: \~65%
* Validation accuracy: \~45%

> Note: Validation accuracy indicates room for improvement (e.g., more data, better augmentation, or fine-tuning the base model).

---

## 🔧 Requirements

```bash
pip install tensorflow keras opencv-python matplotlib scikit-learn
```

---

## 🚀 How to Run

1. Clone the repository
2. Download the dataset from Kaggle and place it in `data/train` and `data/val` folders.
3. Run the training script:

```bash
python train_model.py
```

---

## 🧠 Future Improvements

* Fine-tune MobileNetV2 layers.
* Explore deeper models like EfficientNet or custom CNNs.
* Apply better regularization and more balanced class handling.
* Incorporate temporal features for video-based lie detection.

---

## 📌 Project Structure

```
📁 facial-lie-detection/
├── data/
│   ├── train/
│   └── val/
├── train_model.py
├── model.h5
└── README.md
```



