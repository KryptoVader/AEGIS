# AEGIS Neural Network

A lightweight, modular, from-scratch deep learning and neural network library implemented in pure Java (JDK 21+). Designed for educational clarity, modularity, and seamless integration as an independent Git submodule.

> [!NOTE]
> This is a **pure CPU, from-scratch Java implementation**. It does **not** rely on external tensor libraries (e.g. ND4J, Deeplearning4j, PyTorch, TensorFlow) and does **not** provide GPU support. All matrix algebra, forward passes, backward passes, activation functions, loss functions, and optimizers are custom-built.

---

## Architecture Overview

The library follows a clean, modular structure organized into distinct packages:

| Package | Key Classes / Interfaces | Description |
| :--- | :--- | :--- |
| **`core`** | `Matrix`, `NeuralNetwork` | Core 2D matrix engine with fundamental linear algebra operations and multi-layer feedforward neural network orchestrator. |
| **`layers`** | `Layer`, `Linear` | Fully connected dense layers with forward propagation and analytical gradient backpropagation. |
| **`activations`** | `Activation`, `ReLU`, `Sigmoid`, `Softmax` | Activation functions and their derivatives/Jacobians. |
| **`losses`** | `Loss`, `CrossEntropy`, `MSE` | Loss/cost functions and analytical loss derivatives. |
| **`optimizers`** | `Optimizer`, `SGD` | Parameter optimization algorithms (e.g. Stochastic Gradient Descent with configurable learning rate). |
| **`metrics`** | `Metrics` | Model evaluation utilities, including confusion matrix calculation and classification metrics. |
| **`models`** | `Perceptron`, `LogisticRegression` | Classical machine learning algorithms implemented with the custom `Matrix` engine. |
| **`exceptions`** | `InvalidValue` | Custom exception for matrix dimension mismatches, illegal indexing, and invalid numerical operations. |
| **`examples`** | `MNIST` | End-to-end classification pipeline for the MNIST handwritten digit dataset with real-time visualization. |

---

## Directory Structure

```text
Neural_Network/
├── assets/
│   ├── Confusion_Matrix_MNIST.png
│   ├── Training.png
│   ├── mnist_accuracy.png
│   └── mnist_loss.png
├── pom.xml
├── .gitignore
├── README.md
├── src/
│   └── main/
│       └── java/
│           ├── core/
│           │   ├── Matrix.java
│           │   └── NeuralNetwork.java
│           ├── layers/
│           │   ├── Layer.java
│           │   └── Linear.java
│           ├── activations/
│           │   ├── Activation.java
│           │   ├── ReLU.java
│           │   ├── Sigmoid.java
│           │   └── Softmax.java
│           ├── losses/
│           │   ├── Loss.java
│           │   ├── CrossEntropy.java
│           │   └── MSE.java
│           ├── optimizers/
│           │   ├── Optimizer.java
│           │   └── SGD.java
│           ├── metrics/
│           │   └── Metrics.java
│           ├── models/
│           │   ├── Perceptron.java
│           │   └── LogisticRegression.java
│           └── exceptions/
│               └── InvalidValue.java
└── examples/
    └── MNIST.java
```

---

## Dependencies

- **JDK**: Java 21 LTS or newer.
- **XChart** (`org.knowm.xchart:xchart:3.8.8`): Used exclusively in `examples/MNIST.java` for plotting training loss curves, accuracy curves, and displaying charts via Swing. Managed via Maven Central.

---

## How to Build with Maven

To clean and compile the entire project (including both `src/main/java` and `examples`):

```bash
mvn clean compile
```

To package into a JAR:

```bash
mvn clean package
```

### Direct `javac` Compilation (Without Maven)

If Maven is not installed in your local environment, you can compile the project directly using JDK `javac`:

```powershell
# From the repository root (Neural_Network/):
mkdir -p target/classes

# Compile core library and examples (specifying external XChart jar)
javac -d target/classes -cp "..\lib\xchart-3.8.8.jar" (Get-ChildItem -Recurse -Filter "*.java" src/main/java).FullName examples/MNIST.java
```

---

## Running the MNIST Example

The MNIST example demonstrates training a multi-layer perceptron on the 70,000-image MNIST dataset from scratch.

### 1. Dataset Path
By default, the example looks for the raw MNIST IDX files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`) at:
```
C:\AEGIS\roadmap\Deep_Learning\mnist\data\MNIST\raw
```
You can also pass a custom directory path as a command-line argument.

### 2. Run via Maven

```bash
mvn exec:java -Dexec.mainClass="examples.MNIST"
```

With a custom dataset path:
```bash
mvn exec:java -Dexec.mainClass="examples.MNIST" -Dexec.args="C:\path\to\MNIST\raw"
```

### 3. Run via Java CLI

```powershell
# Default path:
java -cp "target/classes;..\lib\xchart-3.8.8.jar" examples.MNIST

# Custom path:
java -cp "target/classes;..\lib\xchart-3.8.8.jar" examples.MNIST "C:\path\to\MNIST\raw"
```

---

## Current MNIST Benchmark

| Parameter | Value |
| :--- | :--- |
| **Architecture** | `784` (Input) → `128` (Hidden) → `10` (Output) |
| **Hidden Activation** | `ReLU` |
| **Output Activation** | `Softmax` |
| **Loss Function** | `CrossEntropy` |
| **Optimizer** | `SGD` (learning rate = `0.01`) |
| **Batch Size** | `64` |
| **Epochs** | `10` |
| **Final Training Accuracy** | **~90.89%** |
| **Final Test Accuracy** | **~90.11%** |

### Benchmark Visualizations

| Training Loss Curve | Accuracy Curve (Train vs Test) |
| :---: | :---: |
| ![MNIST Training Loss](assets/mnist_loss.png) | ![MNIST Accuracy](assets/mnist_accuracy.png) |

| Execution Logs | Confusion Matrix |
| :---: | :---: |
| ![Training Execution](assets/Training.png) | ![Confusion Matrix](assets/Confusion_Matrix_MNIST.png) |

### Pipeline Features:
1. **Raw IDX Parsing**: Unsigned byte reading and pixel normalization ($[0, 255] \to [0.0, 1.0]$).
2. **One-Hot Encoding**: Converts integer labels ($0-9$) into 10-dimensional unit vectors.
3. **Mini-Batch SGD**: In-place array index shuffling and batch slicing.
4. **Interactive Visualization**: Real-time XChart loss curve and train/test accuracy curve plots.
5. **Confusion Matrix**: Console-formatted 10x10 classification confusion matrix.
