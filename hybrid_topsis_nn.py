"""
Hybrid TOPSIS + Neural Network Model
A combination of TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)
and Neural Network for advanced multi-criteria decision making.

Author: vkas0812
Date: 2025-12-23
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')


class HybridTOPSIS_NN:
    """
    Hybrid model combining TOPSIS algorithm with Neural Network for MCDM.
    
    This model uses TOPSIS to identify ideal and anti-ideal solutions, then
    leverages Neural Networks to learn complex relationships between criteria
    and decision outcomes.
    """
    
    def __init__(self, weights=None, benefit_criteria=None, nn_hidden_layers=(64, 32)):
        """
        Initialize the Hybrid TOPSIS + NN model.
        
        Parameters:
        -----------
        weights : array-like, optional
            Weights for criteria. If None, equal weights are assigned.
        benefit_criteria : array-like, optional
            Boolean array indicating benefit (True) or cost (False) criteria.
            If None, all criteria are assumed to be benefit criteria.
        nn_hidden_layers : tuple
            Hidden layer sizes for the neural network.
        """
        self.weights = weights
        self.benefit_criteria = benefit_criteria
        self.nn_hidden_layers = nn_hidden_layers
        self.scaler = StandardScaler()
        self.topsis_scaler = MinMaxScaler()
        self.nn_model = None
        self.ideal_solution = None
        self.anti_ideal_solution = None
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.n_criteria = None
        
    def calculate_topsis_scores(self, decision_matrix):
        """
        Calculate TOPSIS scores for alternatives.
        
        Parameters:
        -----------
        decision_matrix : array-like, shape (n_alternatives, n_criteria)
            Decision matrix with alternatives as rows and criteria as columns.
            
        Returns:
        --------
        topsis_scores : array, shape (n_alternatives,)
            TOPSIS scores for each alternative.
        """
        decision_matrix = np.array(decision_matrix)
        n_alternatives, n_criteria = decision_matrix.shape
        self.n_criteria = n_criteria
        
        # Initialize benefit criteria if not provided
        if self.benefit_criteria is None:
            self.benefit_criteria = np.ones(n_criteria, dtype=bool)
        
        # Step 1: Normalize the decision matrix (Vector Normalization)
        norm = np.sqrt((decision_matrix ** 2).sum(axis=0))
        self.normalized_matrix = decision_matrix / norm
        
        # Step 2: Calculate weighted normalized matrix
        if self.weights is None:
            self.weights = np.ones(n_criteria) / n_criteria
        else:
            self.weights = np.array(self.weights) / np.sum(self.weights)
        
        self.weighted_matrix = self.normalized_matrix * self.weights
        
        # Step 3: Determine ideal and anti-ideal solutions
        self.ideal_solution = np.zeros(n_criteria)
        self.anti_ideal_solution = np.zeros(n_criteria)
        
        for j in range(n_criteria):
            if self.benefit_criteria[j]:
                self.ideal_solution[j] = self.weighted_matrix[:, j].max()
                self.anti_ideal_solution[j] = self.weighted_matrix[:, j].min()
            else:
                self.ideal_solution[j] = self.weighted_matrix[:, j].min()
                self.anti_ideal_solution[j] = self.weighted_matrix[:, j].max()
        
        # Step 4: Calculate separation measures
        separation_ideal = np.sqrt(((self.weighted_matrix - self.ideal_solution) ** 2).sum(axis=1))
        separation_anti_ideal = np.sqrt(((self.weighted_matrix - self.anti_ideal_solution) ** 2).sum(axis=1))
        
        # Step 5: Calculate TOPSIS scores
        topsis_scores = separation_anti_ideal / (separation_ideal + separation_anti_ideal + 1e-10)
        
        return topsis_scores
    
    def train_nn_component(self, X, y, test_size=0.2, epochs=100, random_state=42):
        """
        Train the Neural Network component.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_criteria)
            Training features (decision matrix).
        y : array-like, shape (n_samples,)
            Target values (decision outcomes).
        test_size : float
            Proportion of data for testing.
        epochs : int
            Maximum number of epochs for training.
        random_state : int
            Random state for reproducibility.
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Determine if regression or classification
        if len(np.unique(y)) <= 10:  # Assume classification for few unique values
            self.nn_model = MLPClassifier(
                hidden_layer_sizes=self.nn_hidden_layers,
                max_iter=epochs,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
            self.nn_model.fit(X_train_scaled, y_train)
            train_score = self.nn_model.score(X_train_scaled, y_train)
            test_score = self.nn_model.score(X_test_scaled, y_test)
            print(f"Neural Network Classifier - Train Accuracy: {train_score:.4f}, Test Accuracy: {test_score:.4f}")
        else:
            self.nn_model = MLPRegressor(
                hidden_layer_sizes=self.nn_hidden_layers,
                max_iter=epochs,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
            self.nn_model.fit(X_train_scaled, y_train)
            train_mse = mean_squared_error(y_train, self.nn_model.predict(X_train_scaled))
            test_mse = mean_squared_error(y_test, self.nn_model.predict(X_test_scaled))
            print(f"Neural Network Regressor - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    
    def predict_hybrid(self, decision_matrix):
        """
        Make predictions using the hybrid TOPSIS + NN model.
        
        Parameters:
        -----------
        decision_matrix : array-like, shape (n_alternatives, n_criteria)
            Decision matrix for prediction.
            
        Returns:
        --------
        predictions : dict
            Dictionary containing:
            - 'topsis_scores': TOPSIS scores
            - 'nn_predictions': Neural Network predictions
            - 'hybrid_scores': Combined scores (average)
        """
        # Calculate TOPSIS scores
        topsis_scores = self.calculate_topsis_scores(decision_matrix)
        
        # Get NN predictions
        X_scaled = self.scaler.transform(decision_matrix)
        if self.nn_model is None:
            raise ValueError("Neural Network model not trained. Call train_nn_component first.")
        
        nn_predictions = self.nn_model.predict(X_scaled)
        
        # Normalize NN predictions to [0, 1] for combining with TOPSIS
        if isinstance(self.nn_model, MLPRegressor):
            nn_scores = (nn_predictions - nn_predictions.min()) / (nn_predictions.max() - nn_predictions.min() + 1e-10)
        else:
            nn_scores = self.nn_model.predict_proba(X_scaled).max(axis=1)
        
        # Combine TOPSIS and NN scores (equal weighting)
        hybrid_scores = (topsis_scores + nn_scores) / 2
        
        return {
            'topsis_scores': topsis_scores,
            'nn_predictions': nn_predictions,
            'nn_scores': nn_scores,
            'hybrid_scores': hybrid_scores
        }
    
    def get_rankings(self, scores):
        """
        Get rankings based on scores.
        
        Parameters:
        -----------
        scores : array-like
            Scores to rank.
            
        Returns:
        --------
        rankings : array
            Rank indices (0-based, where 0 is the best rank).
        """
        return np.argsort(-np.array(scores))


# Example usage
if __name__ == "__main__":
    # Create sample decision matrix
    n_alternatives = 10
    n_criteria = 4
    
    np.random.seed(42)
    decision_matrix = np.random.rand(n_alternatives, n_criteria) * 100
    
    # Define weights and benefit/cost criteria
    weights = [0.3, 0.25, 0.25, 0.2]
    benefit_criteria = [True, True, False, True]  # Third criterion is cost
    
    # Create and test the model
    model = HybridTOPSIS_NN(weights=weights, benefit_criteria=benefit_criteria)
    
    # Generate synthetic target variable for training
    y = np.random.rand(n_alternatives) * 100
    
    print("Training Hybrid TOPSIS + NN Model...")
    model.train_nn_component(decision_matrix, y, epochs=200)
    
    print("\nMaking predictions...")
    results = model.predict_hybrid(decision_matrix)
    
    print("\nTOPSIS Scores:", results['topsis_scores'][:5])
    print("NN Scores:", results['nn_scores'][:5])
    print("Hybrid Scores:", results['hybrid_scores'][:5])
    
    rankings = model.get_rankings(results['hybrid_scores'])
    print("\nTop 3 ranked alternatives (by hybrid score):", rankings[:3])
