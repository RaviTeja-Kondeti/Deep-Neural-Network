# Deep Neural Network from Scratch

![License](https://img.shields.io/badge/License-MIT-green.svg) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg) ![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243.svg) ![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.0%2B-F7931E.svg) ![Accuracy](https://img.shields.io/badge/Accuracy-93%25-brightgreen.svg) ![Status](https://img.shields.io/badge/Status-Active-success.svg)

A fully-functional deep neural network implementation built from scratch using NumPy for binary classification on the Breast Cancer Wisconsin dataset. This project demonstrates understanding of fundamental deep learning concepts including forward propagation, backpropagation, gradient descent, and various optimization techniques.

## 📚 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Performance Metrics](#performance-metrics)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Technologies](#technologies)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

## 🔍 Overview

This project implements a deep neural network classifier from scratch without using high-level deep learning frameworks like TensorFlow or PyTorch. It achieves **93% accuracy** on the Breast Cancer Wisconsin dataset, demonstrating the effectiveness of the implementation.

The network is built using only NumPy for mathematical operations and Scikit-learn for dataset loading, showcasing a deep understanding of the underlying mechanics of neural networks.

## ✨ Key Features

- **From-Scratch Implementation**: Custom implementation of forward and backward propagation without deep learning frameworks
- **Flexible Architecture**: Configurable number of hidden layers and neurons per layer
- **Multiple Activation Functions**: Support for Logistic (Sigmoid) and ReLU activation functions
- **Advanced Weight Initialization**: Xavier and Kaiming (He) initialization for improved convergence
- **Binary Classification**: Optimized for binary classification tasks with logloss cost function
- **Gradient Descent Optimization**: Efficient gradient descent implementation with configurable learning rate
- **Training Visualization**: Cost function visualization across iterations
- **High Accuracy**: Achieves 93% classification accuracy on breast cancer data

## 🏗️ Technical Architecture

### Network Structure

```
Input Layer (30 features)
    ↓
Hidden Layer 1 (15 neurons, ReLU)
    ↓
Hidden Layer 2 (5 neurons, ReLU)
    ↓
Output Layer (1 neuron, Logistic)
```

### Components Implemented

#### 1. **FullyConnectedLayer Class**
- Encapsulates layer properties (weights, biases, activations)
- Supports multiple activation functions
- Implements Xavier/Kaiming weight initialization
- Stores forward and backward pass computations

#### 2. **Activation Functions**
- **Logistic (Sigmoid)**: σ(z) = 1 / (1 + e^(-z))
- **ReLU**: f(z) = max(0, z)

#### 3. **Cost Function**
- Binary Cross-Entropy (Logloss): J = -(1/m) Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]

#### 4. **Optimization**
- Vanilla Gradient Descent with backpropagation
- Configurable learning rate (α = 0.0001)
- 5000 training epochs

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Dataset** | Breast Cancer Wisconsin (Diagnostic) |
| **Training Samples** | 569 samples, 30 features |
| **Architecture** | 30-15-5-1 (Input-Hidden-Hidden-Output) |
| **Activation** | ReLU (Hidden), Logistic (Output) |
| **Learning Rate** | 0.0001 |
| **Epochs** | 5000 |
| **Final Cost** | 0.17 |
| **Classification Accuracy** | **93%** |
| **Training Time** | ~30 seconds |

## 📁 Dataset

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from Scikit-learn:

- **Samples**: 569 instances
- **Features**: 30 numeric features computed from digitized images of breast mass
- **Classes**: 2 (Malignant, Benign)
- **Task**: Binary classification to predict malignancy

### Feature Categories:
1. Radius (mean of distances from center to points on the perimeter)
2. Texture (standard deviation of gray-scale values)
3. Perimeter
4. Area
5. Smoothness (local variation in radius lengths)
6. Compactness (perimeter² / area - 1.0)
7. Concavity (severity of concave portions of the contour)
8. Concave points (number of concave portions of the contour)
9. Symmetry
10. Fractal dimension

Each feature has three measurements: mean, standard error, and worst (largest value).

## 🚀 Installation

### Prerequisites

```bash
Python 3.8 or higher
```

### Required Libraries

```bash
pip install numpy scipy scikit-learn matplotlib
```

### Clone the Repository

```bash
git clone https://github.com/RaviTeja-Kondeti/Deep-Neural-Network.git
cd Deep-Neural-Network
```

## 💻 Usage

### Running the Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook
```

2. Open the main notebook and run all cells

3. The network will:
   - Load the breast cancer dataset
   - Initialize the neural network with specified architecture
   - Train for 5000 epochs
   - Display cost progression
   - Evaluate accuracy on the dataset

### Code Example

```python
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load data
X, Y = load_breast_cancer(return_X_y=True)
Y = Y.reshape(1, -1)

# Initialize network
fcn = FullyConnectedNetwork(
    n0=X.shape[0],      # 30 input features
    nH=(15, 5),         # Hidden layers: 15 and 5 neurons
    nL=1,               # 1 output neuron
    hidden_g='relu',    # ReLU activation for hidden layers
    output_g='logistic' # Logistic activation for output
)

# Train
fcn.fit(X, Y, alpha=0.0001, nepochs=5000)

# Predict
Y_pred = fcn.predict(X)

# Evaluate
accuracy = fcn.accuracy(Y_pred, Y)
print(f"Accuracy: {round(accuracy, 2)}")
```

## 📂 Project Structure

```
Deep-Neural-Network/
│
├── neural_network_implementation.ipynb  # Main implementation notebook
├── README.md                            # Project documentation
└── LICENSE                              # MIT License
```

## 🔧 Implementation Details

### Forward Propagation

For each layer l:
1. Compute linear transformation: Z^[l] = W^[l] · A^[l-1] + b^[l]
2. Apply activation function: A^[l] = g(Z^[l])
3. Store activations for backpropagation

### Backward Propagation

For each layer l (from output to input):
1. Compute activation gradient: dA^[l]
2. Compute gradient of activation function: dZ^[l] = dA^[l] · g'(Z^[l])
3. Compute parameter gradients:
   - dW^[l] = (1/m) · dZ^[l] · A^[l-1]^T
   - db^[l] = (1/m) · Σ dZ^[l]
4. Compute gradient for previous layer: dA^[l-1] = W^[l]^T · dZ^[l]

### Weight Initialization

**Xavier Initialization** (for Logistic/Tanh):
```python
W = np.random.randn(n_out, n_in) * np.sqrt(1/n_in)
```

**Kaiming (He) Initialization** (for ReLU):
```python
W = np.random.randn(n_out, n_in) * np.sqrt(2/n_in)
```

### Helper Functions

| Function | Purpose |
|----------|----------|
| `calc_gZ()` | Compute activation function output |
| `calc_dgZ_dZ()` | Compute activation function gradient |
| `calc_J()` | Calculate binary cross-entropy cost |
| `init_W()` | Initialize weights with Xavier/Kaiming |

## 📈 Results

### Training Progress

The cost function decreases rapidly in the first 500 iterations and converges smoothly:

- **Initial Cost** (iteration 0): ~12.0
- **Cost at iteration 1000**: ~0.5
- **Final Cost** (iteration 5000): **0.17**

### Model Performance

- **Classification Accuracy**: **93%**
- The model successfully identifies malignant and benign tumors with high reliability
- Convergence is stable without overfitting indicators

### Visualization

The implementation generates a cost vs iterations plot showing:
- Rapid initial descent
- Smooth convergence
- Stable final cost around 0.17

## 🔮 Future Enhancements

1. **Advanced Optimization**
   - Implement Adam optimizer
   - Add momentum and RMSProp
   - Learning rate scheduling

2. **Regularization Techniques**
   - L1/L2 regularization
   - Dropout implementation
   - Batch normalization

3. **Performance Improvements**
   - Mini-batch gradient descent
   - Early stopping mechanism
   - Cross-validation

4. **Extended Functionality**
   - Support for multi-class classification
   - Additional activation functions (Tanh, LeakyReLU, ELU)
   - Configurable loss functions

5. **Analysis & Visualization**
   - Confusion matrix
   - ROC curve and AUC score
   - Feature importance analysis
   - Learning curves

## 🛠️ Technologies

- **Python 3.8+**: Core programming language
- **NumPy**: Numerical computations and matrix operations
- **SciPy**: Scientific computing utilities
- **Scikit-learn**: Dataset loading and accuracy metrics
- **Matplotlib**: Visualization of cost function and results
- **Jupyter Notebook**: Interactive development environment

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: Breast Cancer Wisconsin (Diagnostic) dataset from UCI Machine Learning Repository
- **Inspiration**: Deep learning fundamentals from Stanford CS229 and Andrew Ng's Deep Learning Specialization
- **Mathematical Foundations**: Neural Networks and Deep Learning literature

## 📧 Contact

**Ravi Teja Kondeti**

- GitHub: [@RaviTeja-Kondeti](https://github.com/RaviTeja-Kondeti)
- LinkedIn: [Ravi Teja Kondeti](https://www.linkedin.com/in/ravitejakondeti/)

---

<p align="center">
  <i>Built with ❤️ for demonstrating deep learning fundamentals</i>
</p>
