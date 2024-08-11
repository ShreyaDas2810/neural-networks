# Neural Network Framework from Scratch

This repository contains a simple neural network framework implemented from scratch using NumPy. The framework supports core components of a neural network, including layers, activation functions, loss functions, an optimizer, and a model class to encapsulate everything. The project also includes a Kaggle notebook where the framework is used to train a neural network on the MNIST dataset.

## Table of Contents
- [Objective](#objective)
- [Framework Components](#framework-components)
- [Installation](#installation)
- [Usage](#usage)
- [Using the Framework in Kaggle](#using-the-framework-in-kaggle)
- [Evaluation](#evaluation)
- [Kaggle Notebook](#kaggle-notebook)
- [Contributing](#contributing)
- [License](#license)

## Objective

The goal of this project is to create a neural network framework from scratch and train it on the MNIST dataset to achieve at least 80% accuracy on the test set. The framework is implemented in Python using NumPy and is designed to be simple, modular, and easy to use.

## Framework Components

### Layers
- **Linear Layer**: Implements a fully connected (dense) layer.
  - Methods: `forward`, `backward`

### Activation Functions
- **ReLU**: Implements the ReLU activation function.
  - Methods: `forward`, `backward`
- **Sigmoid**: Implements the Sigmoid activation function.
  - Methods: `forward`, `backward`
- **Tanh**: Implements the Tanh activation function.
  - Methods: `forward`, `backward`
- **Softmax**: Implements the Softmax activation function.
  - Methods: `forward`, `backward`

### Loss Functions
- **Cross-Entropy Loss**: Implements the cross-entropy loss function.
  - Methods: `forward`, `backward`
- **Mean Squared Error (MSE)**: Implements the MSE loss function.
  - Methods: `forward`, `backward`

### Optimizer
- **SGD**: Implements the Stochastic Gradient Descent (SGD) optimizer.
  - Methods: `step`
### Link to kaggle notebook
- https://www.kaggle.com/code/dasshr/neural-network/edit
