"""
Data Generator for TOPSIS and Neural Network Comparison
Generates synthetic datasets for multi-criteria decision making evaluation
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class SyntheticDataGenerator:
    """Generate synthetic data for TOPSIS and NN comparison studies"""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the data generator
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def generate_decision_matrix(
        self, 
        n_alternatives: int = 10,
        n_criteria: int = 5,
        value_range: Tuple[float, float] = (1, 100)
    ) -> pd.DataFrame:
        """
        Generate a random decision matrix for TOPSIS analysis
        
        Args:
            n_alternatives: Number of alternatives (rows)
            n_criteria: Number of criteria (columns)
            value_range: Tuple of (min, max) values for the matrix
        
        Returns:
            DataFrame with decision matrix
        """
        data = np.random.uniform(
            value_range[0], 
            value_range[1], 
            (n_alternatives, n_criteria)
        )
        
        columns = [f'Criteria_{i+1}' for i in range(n_criteria)]
        index = [f'Alternative_{i+1}' for i in range(n_alternatives)]
        
        return pd.DataFrame(data, index=index, columns=columns)
    
    def generate_weighted_criteria(self, n_criteria: int = 5) -> np.ndarray:
        """
        Generate random weights for criteria that sum to 1
        
        Args:
            n_criteria: Number of criteria
        
        Returns:
            Array of normalized weights
        """
        weights = np.random.random(n_criteria)
        return weights / weights.sum()
    
    def generate_criterion_types(self, n_criteria: int = 5) -> List[str]:
        """
        Generate random benefit/cost type for each criterion
        
        Args:
            n_criteria: Number of criteria
        
        Returns:
            List of criterion types ('benefit' or 'cost')
        """
        return ['benefit' if np.random.random() > 0.5 else 'cost' 
                for _ in range(n_criteria)]
    
    def generate_training_dataset(
        self,
        n_samples: int = 100,
        n_features: int = 5,
        n_output_classes: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic training data for neural network
        
        Args:
            n_samples: Number of training samples
            n_features: Number of input features
            n_output_classes: Number of output classes
        
        Returns:
            Tuple of (X, y) where X is features and y is labels
        """
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, n_output_classes, n_samples)
        return X, y
    
    def generate_correlation_matrix(
        self,
        n_variables: int = 5,
        correlation_strength: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate a synthetic correlation matrix
        
        Args:
            n_variables: Number of variables
            correlation_strength: Strength of correlation (0-1)
        
        Returns:
            Correlation matrix as DataFrame
        """
        # Generate random data
        data = np.random.randn(100, n_variables)
        
        # Add some correlation structure
        for i in range(n_variables - 1):
            data[:, i+1] += correlation_strength * data[:, i]
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(data.T)
        
        var_names = [f'Var_{i+1}' for i in range(n_variables)]
        return pd.DataFrame(corr_matrix, index=var_names, columns=var_names)
    
    def generate_performance_data(
        self,
        n_iterations: int = 50,
        n_methods: int = 3
    ) -> pd.DataFrame:
        """
        Generate synthetic performance metrics for comparison
        
        Args:
            n_iterations: Number of iterations/runs
            n_methods: Number of methods to compare
        
        Returns:
            DataFrame with performance metrics
        """
        methods = [f'Method_{i+1}' for i in range(n_methods)]
        
        data = {
            method: np.random.uniform(0.6, 0.95, n_iterations)
            for method in methods
        }
        
        df = pd.DataFrame(data)
        df.index.name = 'Iteration'
        return df
    
    def generate_complete_dataset(
        self,
        n_alternatives: int = 15,
        n_criteria: int = 8,
        n_training_samples: int = 150
    ) -> dict:
        """
        Generate a complete dataset for TOPSIS and NN comparison
        
        Args:
            n_alternatives: Number of alternatives
            n_criteria: Number of criteria
            n_training_samples: Number of training samples for NN
        
        Returns:
            Dictionary containing all generated data
        """
        return {
            'decision_matrix': self.generate_decision_matrix(
                n_alternatives, n_criteria
            ),
            'weights': self.generate_weighted_criteria(n_criteria),
            'criterion_types': self.generate_criterion_types(n_criteria),
            'training_data': self.generate_training_dataset(
                n_training_samples, n_criteria, n_alternatives
            ),
            'correlation_matrix': self.generate_correlation_matrix(n_criteria),
            'performance_data': self.generate_performance_data(50, 2)
        }


def main():
    """Main function demonstrating usage"""
    
    # Initialize generator
    generator = SyntheticDataGenerator(random_seed=42)
    
    # Generate individual components
    print("Generating Decision Matrix...")
    dm = generator.generate_decision_matrix(n_alternatives=10, n_criteria=5)
    print(dm)
    print("\n" + "="*50 + "\n")
    
    # Generate weights
    print("Generating Criteria Weights...")
    weights = generator.generate_weighted_criteria(n_criteria=5)
    print(f"Weights: {weights}")
    print(f"Sum of weights: {weights.sum()}")
    print("\n" + "="*50 + "\n")
    
    # Generate criterion types
    print("Generating Criterion Types...")
    types = generator.generate_criterion_types(n_criteria=5)
    print(f"Types: {types}")
    print("\n" + "="*50 + "\n")
    
    # Generate complete dataset
    print("Generating Complete Dataset...")
    complete_data = generator.generate_complete_dataset(
        n_alternatives=15,
        n_criteria=8,
        n_training_samples=150
    )
    
    print("Complete Dataset Keys:")
    for key in complete_data.keys():
        print(f"  - {key}")


if __name__ == "__main__":
    main()
