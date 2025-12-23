"""
TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) Implementation
Author: vkas0812
Date: 2025-12-23
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


class TOPSIS:
    """
    TOPSIS - Technique for Order Preference by Similarity to Ideal Solution
    
    A multi-criteria decision-making (MCDM) method that ranks alternatives
    based on their proximity to the ideal solution and distance from the negative ideal solution.
    """
    
    def __init__(self, decision_matrix: Union[np.ndarray, pd.DataFrame], 
                 weights: Union[list, np.ndarray] = None,
                 impacts: Union[list, str] = None):
        """
        Initialize TOPSIS model.
        
        Parameters:
        -----------
        decision_matrix : np.ndarray or pd.DataFrame
            Decision matrix where rows are alternatives and columns are criteria
        weights : list or np.ndarray, optional
            Weights for each criterion. If None, equal weights are assigned.
        impacts : list or str, optional
            Impact type for each criterion ('+' for benefit, '-' for cost).
            If string, all criteria have same impact type.
        """
        self.decision_matrix = np.array(decision_matrix)
        self.n_alternatives, self.n_criteria = self.decision_matrix.shape
        
        # Initialize weights
        if weights is None:
            self.weights = np.ones(self.n_criteria) / self.n_criteria
        else:
            weights_array = np.array(weights)
            self.weights = weights_array / weights_array.sum()
        
        # Initialize impacts
        if impacts is None:
            self.impacts = ['+'] * self.n_criteria
        elif isinstance(impacts, str):
            self.impacts = [impacts] * self.n_criteria
        else:
            self.impacts = impacts
        
        self.normalized_matrix = None
        self.weighted_matrix = None
        self.ideal_solution = None
        self.negative_ideal_solution = None
        self.separation_positive = None
        self.separation_negative = None
        self.scores = None
        self.rankings = None
    
    def normalize(self) -> np.ndarray:
        """
        Normalize the decision matrix using vector normalization.
        
        Returns:
        --------
        np.ndarray
            Normalized decision matrix
        """
        # Vector normalization: divide each element by the norm of the column
        norms = np.sqrt((self.decision_matrix ** 2).sum(axis=0))
        self.normalized_matrix = self.decision_matrix / norms
        return self.normalized_matrix
    
    def weight_matrix(self) -> np.ndarray:
        """
        Apply weights to normalized matrix.
        
        Returns:
        --------
        np.ndarray
            Weighted normalized decision matrix
        """
        if self.normalized_matrix is None:
            self.normalize()
        
        self.weighted_matrix = self.normalized_matrix * self.weights
        return self.weighted_matrix
    
    def determine_ideal_solutions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine ideal and negative ideal solutions.
        
        Returns:
        --------
        tuple
            (ideal_solution, negative_ideal_solution)
        """
        if self.weighted_matrix is None:
            self.weight_matrix()
        
        self.ideal_solution = np.zeros(self.n_criteria)
        self.negative_ideal_solution = np.zeros(self.n_criteria)
        
        for i in range(self.n_criteria):
            if self.impacts[i] == '+':  # Benefit criterion
                self.ideal_solution[i] = self.weighted_matrix[:, i].max()
                self.negative_ideal_solution[i] = self.weighted_matrix[:, i].min()
            else:  # Cost criterion
                self.ideal_solution[i] = self.weighted_matrix[:, i].min()
                self.negative_ideal_solution[i] = self.weighted_matrix[:, i].max()
        
        return self.ideal_solution, self.negative_ideal_solution
    
    def calculate_separations(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate separation from ideal and negative ideal solutions.
        
        Returns:
        --------
        tuple
            (separation_positive, separation_negative)
        """
        if self.ideal_solution is None:
            self.determine_ideal_solutions()
        
        # Euclidean distance from ideal solution
        self.separation_positive = np.sqrt(
            ((self.weighted_matrix - self.ideal_solution) ** 2).sum(axis=1)
        )
        
        # Euclidean distance from negative ideal solution
        self.separation_negative = np.sqrt(
            ((self.weighted_matrix - self.negative_ideal_solution) ** 2).sum(axis=1)
        )
        
        return self.separation_positive, self.separation_negative
    
    def calculate_scores(self) -> np.ndarray:
        """
        Calculate TOPSIS scores for each alternative.
        
        Returns:
        --------
        np.ndarray
            TOPSIS scores (0 to 1)
        """
        if self.separation_positive is None:
            self.calculate_separations()
        
        # Avoid division by zero
        denominator = self.separation_positive + self.separation_negative
        self.scores = np.where(
            denominator != 0,
            self.separation_negative / denominator,
            0
        )
        
        return self.scores
    
    def rank(self) -> np.ndarray:
        """
        Rank alternatives based on TOPSIS scores.
        
        Returns:
        --------
        np.ndarray
            Ranking of alternatives (1 = best)
        """
        if self.scores is None:
            self.calculate_scores()
        
        self.rankings = np.argsort(-self.scores) + 1  # Sort descending and add 1 for 1-based indexing
        return self.rankings
    
    def solve(self) -> dict:
        """
        Solve TOPSIS and return results.
        
        Returns:
        --------
        dict
            Dictionary containing scores, rankings, and detailed results
        """
        self.normalize()
        self.weight_matrix()
        self.determine_ideal_solutions()
        self.calculate_separations()
        self.calculate_scores()
        self.rank()
        
        # Create rank mapping
        rank_mapping = {i: np.where(self.rankings == i + 1)[0][0] + 1 for i in range(self.n_alternatives)}
        
        results = {
            'scores': self.scores,
            'rankings': self.rankings,
            'rank_mapping': rank_mapping,
            'ideal_solution': self.ideal_solution,
            'negative_ideal_solution': self.negative_ideal_solution,
            'separation_positive': self.separation_positive,
            'separation_negative': self.separation_negative,
            'weighted_matrix': self.weighted_matrix
        }
        
        return results
    
    def get_results_dataframe(self, alternative_names: list = None) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame.
        
        Parameters:
        -----------
        alternative_names : list, optional
            Names for each alternative
        
        Returns:
        --------
        pd.DataFrame
            Results dataframe with alternatives, scores, and rankings
        """
        if self.scores is None:
            self.solve()
        
        if alternative_names is None:
            alternative_names = [f'Alternative_{i+1}' for i in range(self.n_alternatives)]
        
        results_df = pd.DataFrame({
            'Alternative': alternative_names,
            'TOPSIS_Score': self.scores,
            'Rank': self.rankings
        })
        
        results_df = results_df.sort_values('Rank').reset_index(drop=True)
        return results_df


# Example usage
if __name__ == "__main__":
    # Example decision matrix
    # Rows: Alternatives, Columns: Criteria
    decision_matrix = np.array([
        [8, 7, 5, 9],
        [7, 8, 6, 8],
        [9, 6, 7, 7],
        [6, 9, 8, 6]
    ])
    
    # Weights for each criterion
    weights = [0.25, 0.25, 0.25, 0.25]
    
    # Impact types ('+' for benefit, '-' for cost)
    impacts = ['+', '+', '+', '+']
    
    # Create TOPSIS model
    topsis = TOPSIS(decision_matrix, weights, impacts)
    
    # Solve
    results = topsis.solve()
    
    # Get results as dataframe
    alternatives = ['Option_A', 'Option_B', 'Option_C', 'Option_D']
    results_df = topsis.get_results_dataframe(alternatives)
    
    print("TOPSIS Results:")
    print(results_df)
    print("\nScores:", results['scores'])
    print("Rankings:", results['rankings'])
