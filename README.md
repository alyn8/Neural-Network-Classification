# 🧠 MNIST Neural Network Classification

This repository contains an **Artificial Neural Network (ANN / MLP)** implementation built with TensorFlow/Keras to classify handwritten digits (0–9) from the **MNIST** dataset with high accuracy.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/caglakacar/artificial_neural_networks/blob/main/annfinal.ipynb)

---

## 📌 Project Overview

- **Dataset:** MNIST (60,000 Training / 10,000 Test Images)
- **Input Dimensions:** 28x28 Grayscale Images (Flattened to 784-feature vectors)
- **Model Type:** Multi-Layer Perceptron (MLP / Feedforward Neural Network)
- **Architecture:** 2 Hidden Layers + Dropout (Overfitting Prevention)
- **Output Classes:** 10 (Digits 0 through 9)
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Test Accuracy:** ~97.9%

---

## 🛠 Model Architecture

The input images are normalized to the range `[0, 1]` and flattened into 1D vectors of size 784. Target labels are converted into one-hot encoded categorical vectors (10 dimensions).

```text
[Input Layer: 784] 
        │
        ▼
[Dense Layer: 64 Neurons - ReLU]
        │
        ▼
[Dropout Layer: Rate 0.1]
        │
        ▼
[Dense Layer: 64 Neurons - ReLU]
        │
        ▼
[Output Layer: 10 Neurons - Softmax]
```

---

## ⚙️ Hyperparameters

- **Epochs:** 60
- **Batch Size:** 64 (196 when using Learning Rate Decay)
- **Learning Rate:** 0.1
- **Momentum:** 0.8
- **Loss Function:** Categorical Crossentropy
- **LR Scheduler:** Exponential Decay

---

## 📊 Performance & Results

The model was trained for 60 epochs, achieving the following performance metrics:

- **Training Accuracy:** ~98.9%
- **Validation/Test Accuracy:** ~97.9%

*(Loss and accuracy plots across epochs can be found directly inside the Jupyter Notebook outputs.)*

---

## 🚀 Getting Started

Follow these steps to run the notebook locally:

1. **Install Dependencies:**
   ```bash
   pip install tensorflow scikeras matplotlib numpy scikit-learn
   ```

2. **Run the Notebook:**
   Open `annfinal.ipynb` in Jupyter Notebook, VS Code, or Google Colab and run the cells sequentially.

---

## 📂 Requirements

- Python 3.10+
- TensorFlow 2.17+
- Keras 3.5+
- NumPy, Matplotlib, Scikit-Learn
