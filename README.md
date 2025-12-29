# MLP-from-Scratch

Implementation of a Multi-Layer Perceptron (MLP) **fully from scratch** using NumPy.  

## Features

- Manual forward and backward passes
- Layers implemented from scratch:  
  - `Linear`  
  - `Embedding`  
  - `ReLU`  
  - `SoftmaxCrossEntropy` (numerically stable)  
  - `Flatten`  
  - `MLP` container for sequential layers
- Parameter updates done manually with gradient descent

## Experiments

### 1. XOR Sanity Check
- File: `train_xor.ipynb`  
- A sanity check for non-linearity of the model 
- Ensures MLP can solve simple non-linear problems

### 2. MNIST Handwritten Digit Recognition
- File: `train_mnist.ipynb`  
- Achieves **~93% accuracy** on MNIST  
- Demonstrates ability to scale MLP to real datasets

### 3. Next-Token Prediction (Tiny Shakespeare)
- File: `train_tiny.ipynb`  
- Dataset: `input.txt`  
- Trains a simple character-level language model for next-token prediction
- Shows some local linguistic correctness (this, he, she)
