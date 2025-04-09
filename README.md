# Neural Network from Scratch

A simple implementation of a feedforward neural network built from scratch using NumPy. This project demonstrates the fundamental concepts of neural networks including forward propagation, backward propagation, and gradient descent.

## Features

- Simple neural network with one hidden layer
- Sigmoid activation function
- Binary cross-entropy loss
- Configurable input, hidden, and output layer sizes
- Batch gradient descent optimization

## Requirements

- Python 3.x
- NumPy

## Usage

```python
from neural_network import NeuralNetwork

# Initialize the network
nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# Train the network
nn.train(X_train, y_train, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.forward(X_test)
```

## Model Architecture

- Input Layer: Configurable size
- Hidden Layer: Configurable size with sigmoid activation
- Output Layer: Configurable size with sigmoid activation

## Implementation Details

The network implements:
- Weight initialization with small random values
- Forward propagation with sigmoid activation
- Backward propagation with gradient descent
- Binary cross-entropy loss calculation

## License

MIT License
