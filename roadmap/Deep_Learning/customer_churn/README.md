# Customer Churn Prediction — PyTorch ANN Implementation

This repository contains the PyTorch Artificial Neural Network (ANN) implementation for predicting bank customer churn using the `Churn_Modelling.csv` dataset in [`ann.ipynb`](file:///c:/AEGIS/roadmap/Deep_Learning/customer_churn/ann.ipynb).

---

## 📊 Dataset Overview

- **Path**: [`data/Churn_Modelling.csv`](file:///c:/AEGIS/roadmap/Deep_Learning/customer_churn/data/Churn_Modelling.csv)
- **Total Records**: 10,000 customers (8,000 train / 2,000 test split)
- **Target Variable**: `Exited` (Binary: `0` = Stayed [1,607 in test], `1` = Exited [393 in test])

### Feature Engineering & Preprocessing
Non-predictive columns (`RowNumber`, `CustomerId`, `Surname`) were removed, leaving **11 input features**:

1. **Numerical Features (StandardScaler)**: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.
2. **Categorical Features (OneHotEncoder with `drop='first'`)**: 
   - `Geography` (`France`, `Germany`, `Spain`)
   - `Gender` (`Female`, `Male`)

---

## 🏗️ PyTorch Architecture & Training Pipeline

### 1. Custom Dataset & DataLoaders
- Custom `Dataset` class converting NumPy arrays to `torch.float32` features and `torch.long` targets.
- Batched with `DataLoader(batch_size=1000)` on GPU (`cuda`).

### 2. Neural Network Topology (`ReLU` Activation)
```python
ANN(
  (model): Sequential(
    (0): Linear(in_features=11, out_features=5)
    (1): ReLU()
    (2): Linear(in_features=5, out_features=3)
    (3): ReLU()
    (4): Linear(in_features=3, out_features=2)
  )
)
```

### 3. Hyperparameters
- **Loss Function**: `nn.CrossEntropyLoss()`
- **Optimizer**: `optim.SGD(lr=0.5)`
- **Epochs**: 100

---

## 📈 Improved Model Results

### Classification Report (Test Set)
```text
              precision    recall  f1-score   support

           0       0.88      0.96      0.92      1607
           1       0.74      0.49      0.59       393

    accuracy                           0.87      2000
   macro avg       0.81      0.72      0.76      2000
weighted avg       0.86      0.87      0.86      2000
```

---

## 💡 Key Improvements & Analysis

1. **Activation Upgrade (`Sigmoid` ➔ `ReLU`)**:
   - Switching from `Sigmoid` to `ReLU` in hidden layers eliminated vanishing gradients.
   - Training loss dropped faster to **`0.344` in 100 epochs** (versus `0.403` after 500 epochs with Sigmoid).
2. **Significant Metric Gains**:
   - **Overall Accuracy**: **`87%`** (up from `83%`).
   - **Churner Recall (Class 1)**: **`0.49`** (nearly doubled from `0.27`).
   - **Churner Precision (Class 1)**: **`0.74`** (up from `0.66`).
   - **Churner F1-Score (Class 1)**: **`0.59`** (up from `0.38`).
3. **Next Frontier**:
   - Apply class weights in `nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]))` or use `optim.Adam` to boost churner recall past `0.75+`.
