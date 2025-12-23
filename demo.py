"""
Demo script for Hybrid TOPSIS + Neural Network Model
======================================================

This script demonstrates the integration of TOPSIS (Technique for Order 
Preference by Similarity to Ideal Solution) with a Neural Network for 
multi-criteria decision making on synthetic data.

Date: 2025-12-23
Author: vkas0812
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TOPSIS:
    """
    TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
    implementation for multi-criteria decision making.
    """
    
    def __init__(self, weights=None, impacts=None):
        """
        Initialize TOPSIS model.
        
        Args:
            weights: List of weights for each criterion (will be normalized)
            impacts: List of '+' (benefit) or '-' (cost) for each criterion
        """
        self.weights = weights
        self.impacts = impacts
        self.decision_matrix = None
        self.normalized_matrix = None
        self.weighted_matrix = None
        
    def fit(self, decision_matrix):
        """
        Fit the TOPSIS model on decision matrix.
        
        Args:
            decision_matrix: numpy array of shape (alternatives, criteria)
        """
        self.decision_matrix = np.array(decision_matrix, dtype=float)
        
        # Normalize the decision matrix
        self._normalize()
        
        # Apply weights
        self._apply_weights()
        
        return self
    
    def _normalize(self):
        """Normalize decision matrix using vector normalization."""
        sum_squares = np.sqrt((self.decision_matrix ** 2).sum(axis=0))
        self.normalized_matrix = self.decision_matrix / sum_squares
    
    def _apply_weights(self):
        """Apply weights to normalized matrix."""
        if self.weights is None:
            self.weights = np.ones(self.decision_matrix.shape[1])
        
        weights = np.array(self.weights)
        weights = weights / weights.sum()  # Normalize weights
        self.weighted_matrix = self.normalized_matrix * weights
    
    def predict(self):
        """
        Calculate TOPSIS scores for each alternative.
        
        Returns:
            scores: TOPSIS scores for each alternative
            ranks: Ranking of alternatives (1 = best)
        """
        # Determine ideal and anti-ideal solutions
        ideal_solution = np.zeros(self.weighted_matrix.shape[1])
        anti_ideal_solution = np.zeros(self.weighted_matrix.shape[1])
        
        if self.impacts is None:
            self.impacts = ['+'] * self.weighted_matrix.shape[1]
        
        for i, impact in enumerate(self.impacts):
            if impact == '+':  # Benefit criterion
                ideal_solution[i] = self.weighted_matrix[:, i].max()
                anti_ideal_solution[i] = self.weighted_matrix[:, i].min()
            else:  # Cost criterion
                ideal_solution[i] = self.weighted_matrix[:, i].min()
                anti_ideal_solution[i] = self.weighted_matrix[:, i].max()
        
        # Calculate separation measures
        separation_ideal = np.sqrt(((self.weighted_matrix - ideal_solution) ** 2).sum(axis=1))
        separation_anti_ideal = np.sqrt(((self.weighted_matrix - anti_ideal_solution) ** 2).sum(axis=1))
        
        # Calculate TOPSIS scores
        scores = separation_anti_ideal / (separation_ideal + separation_anti_ideal + 1e-10)
        
        # Rank alternatives (higher score = better rank)
        ranks = np.argsort(-scores) + 1
        
        return scores, ranks


class HybridTOPSISNN:
    """
    Hybrid model combining TOPSIS for feature ranking and Neural Network for classification.
    """
    
    def __init__(self, topsis_weights=None, topsis_impacts=None, 
                 nn_hidden_layers=(64, 32), nn_max_iter=200):
        """
        Initialize Hybrid TOPSIS + NN model.
        
        Args:
            topsis_weights: Weights for TOPSIS criteria
            topsis_impacts: Impacts for TOPSIS criteria
            nn_hidden_layers: Hidden layer sizes for NN
            nn_max_iter: Maximum iterations for NN training
        """
        self.topsis = TOPSIS(weights=topsis_weights, impacts=topsis_impacts)
        self.scaler = StandardScaler()
        self.nn = MLPClassifier(hidden_layer_sizes=nn_hidden_layers, 
                               max_iter=nn_max_iter, random_state=42)
        self.topsis_scores = None
    
    def fit(self, X_train, y_train):
        """
        Fit the hybrid model.
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels
        """
        # Calculate TOPSIS scores for feature importance
        self.topsis.fit(X_train)
        self.topsis_scores, _ = self.topsis.predict()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train neural network
        self.nn.fit(X_train_scaled, y_train)
        
        return self
    
    def predict(self, X_test):
        """
        Predict on test data.
        
        Args:
            X_test: Test features
            
        Returns:
            predictions: Predicted labels
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.nn.predict(X_test_scaled)
    
    def predict_proba(self, X_test):
        """
        Predict probabilities for test data.
        
        Args:
            X_test: Test features
            
        Returns:
            probabilities: Predicted probabilities
        """
        X_test_scaled = self.scaler.transform(X_test)
        return self.nn.predict_proba(X_test_scaled)


def generate_synthetic_data(n_samples=200, n_features=5, n_classes=3, random_state=42):
    """
    Generate synthetic dataset for demonstration.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed for reproducibility
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        feature_names: Names of features
    """
    np.random.seed(random_state)
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features) * 10 + 50
    
    # Create target based on feature combinations
    y = (X[:, 0] + X[:, 1] > 100).astype(int)
    y = np.where(X[:, 2] > 50, y + 1, y)
    y = np.clip(y, 0, n_classes - 1)
    
    feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    return X, y, feature_names


def main():
    """Main execution function for demo."""
    
    print("=" * 70)
    print("Hybrid TOPSIS + Neural Network Model Demo")
    print("=" * 70)
    print()
    
    # Generate synthetic data
    print("Step 1: Generating synthetic data...")
    X, y, feature_names = generate_synthetic_data(n_samples=300, n_features=5, n_classes=3)
    print(f"  - Data shape: {X.shape}")
    print(f"  - Number of classes: {len(np.unique(y))}")
    print(f"  - Class distribution: {np.bincount(y)}")
    print()
    
    # Split data
    print("Step 2: Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"  - Training set size: {X_train.shape[0]}")
    print(f"  - Test set size: {X_test.shape[0]}")
    print()
    
    # Initialize and train hybrid model
    print("Step 3: Training Hybrid TOPSIS + NN Model...")
    topsis_weights = [1, 1, 1, 1, 1]  # Equal weights for all features
    topsis_impacts = ['+', '+', '+', '+', '+']  # All benefit criteria
    
    model = HybridTOPSISNN(
        topsis_weights=topsis_weights,
        topsis_impacts=topsis_impacts,
        nn_hidden_layers=(64, 32),
        nn_max_iter=300
    )
    
    model.fit(X_train, y_train)
    print("  - Model training completed")
    print()
    
    # Feature importance from TOPSIS
    print("Step 4: Feature Importance (from TOPSIS):")
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'TOPSIS_Score': model.topsis_scores
    }).sort_values('TOPSIS_Score', ascending=False)
    print(feature_importance.to_string(index=False))
    print()
    
    # Make predictions
    print("Step 5: Making predictions on test set...")
    y_pred = model.predict(X_test)
    print(f"  - Predictions shape: {y_pred.shape}")
    print()
    
    # Evaluate model
    print("Step 6: Model Evaluation:")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  - Accuracy: {accuracy:.4f}")
    print()
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    
    # Confusion matrix
    print("Step 7: Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print()
    
    # Visualize results
    print("Step 8: Generating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Feature Importance
    ax = axes[0, 0]
    feature_importance_sorted = feature_importance.sort_values('TOPSIS_Score')
    ax.barh(feature_importance_sorted['Feature'], feature_importance_sorted['TOPSIS_Score'])
    ax.set_xlabel('TOPSIS Score')
    ax.set_title('Feature Importance (TOPSIS Scores)')
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Confusion Matrix Heatmap
    ax = axes[0, 1]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    
    # Plot 3: Prediction Probability Distribution
    ax = axes[1, 0]
    y_proba = model.predict_proba(X_test)
    max_proba = y_proba.max(axis=1)
    ax.hist(max_proba, bins=20, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Prediction Confidence')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Model Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Model Performance Summary
    ━━━━━━━━━━━━━━━━━━━━━━━━
    
    Accuracy:        {accuracy:.4f}
    Training Samples: {len(X_train)}
    Test Samples:    {len(X_test)}
    
    Number of Features: {X.shape[1]}
    Number of Classes:  {len(np.unique(y))}
    
    NN Architecture: {[X.shape[1], 64, 32, len(np.unique(y))]}
    """
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
            verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('hybrid_topsis_nn_results.png', dpi=300, bbox_inches='tight')
    print("  - Visualization saved as 'hybrid_topsis_nn_results.png'")
    plt.show()
    
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
