# MNIST Handwritten Digit Classification — PyTorch ANN Implementation

This document summarizes the PyTorch Artificial Neural Network (ANN) implementation for classifying handwritten digits ($0-9$) on the **MNIST Dataset** in [`ann.ipynb`](file:///c:/AEGIS/roadmap/Deep_Learning/mnist/ann.ipynb).

---

## 📁 Dataset & Preprocessing

- **Dataset Source**: Loaded directly via `torchvision.datasets.MNIST`.
- **Training Set**: 60,000 grayscale images ($28 \times 28$ pixels).
- **Test Set**: 10,000 grayscale images ($28 \times 28$ pixels).
- **Normalization**: Pixel values scaled from $[0, 255]$ down to $[0.0, 1.0]$ via division by `255.0`.
- **Custom Dataset & DataLoaders**: 
  - Custom `MyDataSet` wrapper mapping image tensors and integer class target labels.
  - Training mini-batch size of `64` (shuffled).
  - Test mini-batch size of `1000`.

---

## 🏗️ Model Architecture (`MyANN`)

The network is a Multilayer Perceptron (MLP) built with `nn.Sequential`:

1. **Flatten Layer**: Reshapes 2D images from $28 \times 28$ into a 1D vector of $784$ features.
2. **Hidden Layer 1**: Fully connected linear layer (`784` $\rightarrow$ `128` neurons) with non-linear `ReLU` activation.
3. **Hidden Layer 2**: Fully connected linear layer (`128` $\rightarrow$ `64` neurons) with non-linear `ReLU` activation.
4. **Output Layer**: Fully connected linear layer (`64` $\rightarrow$ `10` output logits corresponding to digit classes $0$ to $9$).

---

## ⚙️ Hyperparameters & Training Configuration

- **Device Acceleration**: Executed on GPU (`cuda`).
- **Loss Function**: `nn.CrossEntropyLoss()` (computes softmax internally for multi-class classification).
- **Optimizer**: `optim.Adam` with a learning rate of `0.001`.
- **Training Duration**: 10 Epochs.

---

## 📈 Model Performance & Evaluation Results

Evaluated on the 10,000 unseen test samples using `classification_report`:

- **Overall Accuracy**: **98%**
- **Macro Average F1-Score**: **0.98**
- **Weighted Average F1-Score**: **0.98**
- **Class-wise Precision & Recall**: Ranged consistently between **97% and 99%** across all digit classes (`0` through `9`).
