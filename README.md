## ChestXRay-Pneumonia-Detection

This project implements a deep learning pipeline for classifying chest X-ray images to detect **Pneumonia** using a pre-trained **DenseNet121** model. It leverages transfer learning to fine-tune the model on the dataset and evaluate performance with metrics such as classification report and confusion matrix.

---

## 📂 Project Structure

```
.
├── train_densenet.py      # Main training script
├── requirements.txt       # Required dependencies
├── README.md              # Project documentation
└── data/                  # Dataset folder (after extraction)
    ├── train/             # Training images
    ├── val/               # Validation images
    └── test/              # Test images
```

---

## ⚙️ Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. **Download the dataset** from Kaggle:  
   [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

2. **Extract the dataset** and place it inside the project folder as:

```
data/
    train/
    val/
    test/
```

3. **Run the training script**:

```bash
python train_densenet.py
```

---

## 📊 Model & Evaluation

- **Base Model**: DenseNet121 (pre-trained on ImageNet)
- **Task**: Binary classification (Pneumonia vs Normal)
- **Evaluation Metrics**:
  - Accuracy
  - Classification Report (Precision, Recall, F1-score)
  - Confusion Matrix

After training, the script will print evaluation results on the **test dataset**.


---

## 📌 Notes
 
- Ensure GPU is available for faster training.  

---

## ✅ Quick Start

```bash
git clone https://github.com/RonakAdimi/ChestXRay-Pneumonia-Detection.git
cd ChestXRay-Pneumonia-Detection
pip install -r requirements.txt
python train_densenet.py
```

---
