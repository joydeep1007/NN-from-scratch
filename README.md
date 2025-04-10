# Neural Network from Scratch

A simple implementation of a feedforward neural network built from scratch using NumPy. This project demonstrates the fundamental concepts of neural networks including forward propagation, backward propagation, and gradient descent.

## Features

- Neural network with one hidden layer and momentum
- He weight initialization for better convergence
- Sigmoid activation function
- Binary cross-entropy loss
- Configurable input, hidden, and output layer sizes
- Momentum-based gradient descent optimization
- Weight saving and loading functionality

## Requirements

- Python 3.x
- NumPy

## Usage

```python
from neural_network import NeuralNetwork

# Initialize the network with improved architecture
nn = NeuralNetwork(input_size=2, hidden_size=8, output_size=1)

# Train the network with optimized parameters
nn.train(X_train, y_train, epochs=2000, learning_rate=0.05)

# Make predictions
predictions = nn.forward(X_test)

# Save model weights
np.savez('model_weights.npz', 
         W1=nn.W1, 
         W2=nn.W2, 
         b1=nn.b1, 
         b2=nn.b2)
```

## Model Architecture

- Input Layer: Configurable size
- Hidden Layer: Enhanced with 8 neurons (default) and momentum
- Output Layer: Configurable size with sigmoid activation
- Weight Initialization: He initialization for better training

## Implementation Details

The network implements:
- He weight initialization for better gradient flow
- Momentum-based gradient descent for faster convergence
- Forward propagation with sigmoid activation
- Backward propagation with momentum updates
- Binary cross-entropy loss calculation
- Progress monitoring with loss printing every 100 epochs
- Interactive prediction interface for testing

## Example Application

The implementation includes a complete solution for the XOR problem:
- Training accuracy typically reaches 100%
- Interactive testing interface
- Supports binary input (0,1) predictions
- Model weight persistence

## License

MIT License
