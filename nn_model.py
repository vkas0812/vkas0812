"""
Neural Network implementation for Multi-Criteria Decision Making (MCDM)
This module provides neural network models for MCDM applications.
"""

import numpy as np
from typing import Tuple, List, Optional
import warnings


class NeuralNetworkMCDM:
    """
    A neural network model designed for Multi-Criteria Decision Making (MCDM) tasks.
    
    This network can process multiple criteria and alternatives to support decision-making.
    """
    
    def __init__(self, input_size: int, hidden_layers: List[int], output_size: int, 
                 learning_rate: float = 0.01, activation: str = 'relu'):
        """
        Initialize the Neural Network for MCDM.
        
        Args:
            input_size: Number of input features (criteria)
            hidden_layers: List of hidden layer sizes
            output_size: Number of output nodes (alternatives/scores)
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('relu' or 'sigmoid')
        """
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self._initialize_parameters()
        
        self.loss_history = []
        
    def _initialize_parameters(self):
        """Initialize network weights and biases using Xavier initialization."""
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation_function(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def _activation_derivative(self, x: np.ndarray) -> np.ndarray:
        """Calculate derivative of activation function."""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
            
        Returns:
            Tuple of (output, activations) where activations stores all layer outputs
        """
        activations = [X]
        A = X
        
        for i in range(len(self.weights) - 1):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self._activation_function(Z)
            activations.append(A)
        
        # Output layer (linear for regression, softmax for classification)
        Z = np.dot(A, self.weights[-1]) + self.biases[-1]
        output = Z  # Linear output for MCDM scoring
        activations.append(output)
        
        return output, activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, 
                 activations: List[np.ndarray]):
        """
        Backward propagation to calculate gradients.
        
        Args:
            X: Input data
            y: Target values
            output: Network output
            activations: Activations from forward pass
        """
        m = X.shape[0]
        
        # Output layer error
        delta = (output - y) / m
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            dW = np.dot(activations[i].T, delta)
            db = np.sum(delta, axis=0, keepdims=True)
            
            # Update parameters
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # Calculate delta for previous layer
            if i > 0:
                delta = np.dot(delta, self.weights[i].T)
                Z = np.dot(activations[i - 1], self.weights[i - 1]) + self.biases[i - 1]
                delta *= self._activation_derivative(Z)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, 
              batch_size: int = 32, verbose: bool = True):
        """
        Train the neural network.
        
        Args:
            X: Training input data
            y: Training target data
            epochs: Number of training epochs
            batch_size: Size of mini-batches
            verbose: Whether to print loss during training
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward pass
                output, activations = self.forward(X_batch)
                
                # Calculate loss (MSE)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                
                # Backward pass
                self.backward(X_batch, y_batch, output, activations)
            
            self.loss_history.append(epoch_loss)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        output, _ = self.forward(X)
        return output
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate model performance using MSE.
        
        Args:
            X: Evaluation input data
            y: Evaluation target data
            
        Returns:
            Mean squared error
        """
        output, _ = self.forward(X)
        mse = np.mean((output - y) ** 2)
        return mse


class MCDMPreprocessor:
    """Preprocessing utilities for MCDM data."""
    
    @staticmethod
    def normalize(X: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
        """
        Normalize input data.
        
        Args:
            X: Input data
            method: Normalization method ('minmax' or 'zscore')
            
        Returns:
            Normalized data and normalization parameters
        """
        if method == 'minmax':
            min_vals = np.min(X, axis=0)
            max_vals = np.max(X, axis=0)
            X_normalized = (X - min_vals) / (max_vals - min_vals + 1e-8)
            return X_normalized, {'min': min_vals, 'max': max_vals, 'method': 'minmax'}
        
        elif method == 'zscore':
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
            X_normalized = (X - mean) / (std + 1e-8)
            return X_normalized, {'mean': mean, 'std': std, 'method': 'zscore'}
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def apply_normalization(X: np.ndarray, params: dict) -> np.ndarray:
        """Apply saved normalization parameters to new data."""
        if params['method'] == 'minmax':
            return (X - params['min']) / (params['max'] - params['min'] + 1e-8)
        elif params['method'] == 'zscore':
            return (X - params['mean']) / (params['std'] + 1e-8)


# Example usage
if __name__ == "__main__":
    # Create synthetic MCDM dataset
    np.random.seed(42)
    n_samples = 200
    n_criteria = 5
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_criteria)
    y = np.sum(X[:, :3], axis=1, keepdims=True) + 0.5 * np.random.randn(n_samples, 1)
    
    # Normalize data
    preprocessor = MCDMPreprocessor()
    X_norm, norm_params = preprocessor.normalize(X, method='minmax')
    
    # Create and train model
    model = NeuralNetworkMCDM(
        input_size=n_criteria,
        hidden_layers=[32, 16],
        output_size=1,
        learning_rate=0.01,
        activation='relu'
    )
    
    print("Training Neural Network for MCDM...")
    model.train(X_norm, y, epochs=100, batch_size=32, verbose=True)
    
    # Make predictions
    predictions = model.predict(X_norm[:10])
    print(f"\nSample predictions: {predictions.flatten()[:5]}")
    
    # Evaluate
    mse = model.evaluate(X_norm, y)
    print(f"\nFinal MSE: {mse:.6f}")
