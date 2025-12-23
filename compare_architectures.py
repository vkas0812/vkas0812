"""
Comprehensive Neural Network Architecture Comparison for TOPSIS + NN Hybrid Model

This script compares multiple neural network architectures for integration with TOPSIS
(Technique for Order Preference by Similarity to Ideal Solution) in a hybrid model.
Includes synthetic data generation, model training, evaluation, and visualization.

Author: vkas0812
Date: 2025-12-23
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
import time
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class SyntheticDataGenerator:
    """Generate synthetic multi-criteria decision-making data for testing."""
    
    def __init__(self, n_samples: int = 1000, n_criteria: int = 8, noise_level: float = 0.1):
        """
        Initialize data generator.
        
        Args:
            n_samples: Number of samples to generate
            n_criteria: Number of criteria/features
            noise_level: Standard deviation of noise to add
        """
        self.n_samples = n_samples
        self.n_criteria = n_criteria
        self.noise_level = noise_level
        
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic MCDM data.
        
        Returns:
            Tuple of (features, targets)
        """
        # Generate synthetic criteria features
        X = np.random.uniform(0, 100, (self.n_samples, self.n_criteria))
        
        # Create target based on weighted combination of criteria
        # Simulating TOPSIS scores that a NN would learn to enhance
        weights = np.random.dirichlet(np.ones(self.n_criteria))
        y = np.dot(X, weights) + np.random.normal(0, self.noise_level * 100, self.n_samples)
        y = np.clip(y, 0, 100)  # Keep scores in realistic range
        
        return X, y.reshape(-1, 1)
    
    def get_normalized_data(self) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        """
        Generate and normalize data.
        
        Returns:
            Tuple of (normalized_X, normalized_y, scaler)
        """
        X, y = self.generate_data()
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        y_scaled = (y - y.min()) / (y.max() - y.min())
        
        return X_scaled, y_scaled, scaler


class NNArchitectures:
    """Collection of different neural network architectures for comparison."""
    
    @staticmethod
    def simple_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        Simple Multi-Layer Perceptron (MLP).
        
        Architecture: Dense(64) -> Dense(32) -> Dense(output_dim)
        """
        model = models.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(output_dim, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def deep_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        Deep Multi-Layer Perceptron with more layers.
        
        Architecture: Dense(128) -> Dense(64) -> Dense(32) -> Dense(16) -> Dense(output_dim)
        """
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(output_dim, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def wide_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        Wide Multi-Layer Perceptron with large hidden layers.
        
        Architecture: Dense(256) -> Dense(128) -> Dense(output_dim)
        """
        model = models.Sequential([
            layers.Dense(256, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(output_dim, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def residual_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        MLP with Residual Connections.
        
        Includes skip connections for improved gradient flow.
        """
        inputs = layers.Input(shape=(input_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x_residual = x
        
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, x_residual])
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x_residual2 = x
        
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, x_residual2])
        
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(output_dim, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model
    
    @staticmethod
    def bottleneck_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        Bottleneck MLP with compression and expansion.
        
        Architecture: Dense(128) -> Dense(32) -> Dense(128) -> Dense(output_dim)
        """
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),  # Bottleneck
            layers.BatchNormalization(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(output_dim, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def dropout_heavy_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        MLP with heavy dropout for regularization.
        
        Useful for preventing overfitting with noisy MCDM data.
        """
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_dim=input_dim),
            layers.Dropout(0.4),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(output_dim, activation='sigmoid')
        ])
        return model
    
    @staticmethod
    def ensemble_style_mlp(input_dim: int, output_dim: int = 1) -> keras.Model:
        """
        Multi-branch MLP that concatenates multiple pathways.
        
        Inspired by ensemble methods, uses multiple parallel branches.
        """
        inputs = layers.Input(shape=(input_dim,))
        
        # Branch 1: Wide and shallow
        branch1 = layers.Dense(128, activation='relu')(inputs)
        branch1 = layers.Dropout(0.2)(branch1)
        
        # Branch 2: Deep
        branch2 = layers.Dense(64, activation='relu')(inputs)
        branch2 = layers.Dense(32, activation='relu')(branch2)
        branch2 = layers.Dense(32, activation='relu')(branch2)
        
        # Branch 3: Bottleneck
        branch3 = layers.Dense(32, activation='relu')(inputs)
        branch3 = layers.Dense(16, activation='relu')(branch3)
        branch3 = layers.Dense(32, activation='relu')(branch3)
        
        # Concatenate branches
        concat = layers.Concatenate()([branch1, branch2, branch3])
        x = layers.Dense(64, activation='relu')(concat)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(output_dim, activation='sigmoid')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        return model


class ArchitectureComparison:
    """Compare multiple neural network architectures."""
    
    def __init__(self, X_train: np.ndarray, X_test: np.ndarray, 
                 y_train: np.ndarray, y_test: np.ndarray):
        """
        Initialize comparison framework.
        
        Args:
            X_train, X_test: Training and test features
            y_train, y_test: Training and test targets
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.results = {}
        self.histories = {}
        
    def train_architecture(self, name: str, model: keras.Model, 
                          epochs: int = 100, batch_size: int = 32,
                          optimizer: str = 'adam') -> Dict:
        """
        Train a single architecture.
        
        Args:
            name: Architecture name
            model: Keras model to train
            epochs: Number of training epochs
            batch_size: Batch size for training
            optimizer: Optimizer to use
            
        Returns:
            Dictionary with training metrics
        """
        print(f"\nTraining {name}...")
        
        # Compile model
        if optimizer == 'adam':
            opt = Adam(learning_rate=0.001)
        elif optimizer == 'sgd':
            opt = SGD(learning_rate=0.01, momentum=0.9)
        else:
            opt = RMSprop(learning_rate=0.001)
            
        model.compile(optimizer=opt, loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, 
                                       restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, 
                                      patience=5, min_lr=1e-6)
        
        # Train
        start_time = time.time()
        history = model.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        training_time = time.time() - start_time
        
        # Evaluate
        train_loss, train_mae = model.evaluate(self.X_train, self.y_train, verbose=0)
        test_loss, test_mae = model.evaluate(self.X_test, self.y_test, verbose=0)
        
        y_pred = model.predict(self.X_test, verbose=0)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(self.y_test, y_pred)
        
        # Model complexity
        params = model.count_params()
        
        result = {
            'model': model,
            'train_loss': train_loss,
            'train_mae': train_mae,
            'test_loss': test_loss,
            'test_mae': test_mae,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'params': params,
            'training_time': training_time,
            'history': history
        }
        
        self.results[name] = result
        self.histories[name] = history
        
        print(f"{name} - RMSE: {rmse:.4f}, R²: {r2:.4f}, Params: {params}, Time: {training_time:.2f}s")
        
        return result
    
    def compare_all(self, epochs: int = 100, batch_size: int = 32) -> pd.DataFrame:
        """
        Train and compare all architectures.
        
        Returns:
            DataFrame with comparison results
        """
        architectures = {
            'Simple MLP': NNArchitectures.simple_mlp(self.X_train.shape[1]),
            'Deep MLP': NNArchitectures.deep_mlp(self.X_train.shape[1]),
            'Wide MLP': NNArchitectures.wide_mlp(self.X_train.shape[1]),
            'Residual MLP': NNArchitectures.residual_mlp(self.X_train.shape[1]),
            'Bottleneck MLP': NNArchitectures.bottleneck_mlp(self.X_train.shape[1]),
            'Dropout Heavy MLP': NNArchitectures.dropout_heavy_mlp(self.X_train.shape[1]),
            'Ensemble Style MLP': NNArchitectures.ensemble_style_mlp(self.X_train.shape[1])
        }
        
        for name, model in architectures.items():
            self.train_architecture(name, model, epochs=epochs, batch_size=batch_size)
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Architecture': list(self.results.keys()),
            'RMSE': [self.results[k]['rmse'] for k in self.results.keys()],
            'MAE': [self.results[k]['mae'] for k in self.results.keys()],
            'R² Score': [self.results[k]['r2'] for k in self.results.keys()],
            'Parameters': [self.results[k]['params'] for k in self.results.keys()],
            'Training Time (s)': [self.results[k]['training_time'] for k in self.results.keys()],
            'Test MSE': [self.results[k]['mse'] for k in self.results.keys()]
        })
        
        comparison_df = comparison_df.sort_values('RMSE')
        
        return comparison_df
    
    def plot_performance_comparison(self, metric: str = 'RMSE'):
        """Plot performance comparison across architectures."""
        results_df = pd.DataFrame({
            'Architecture': list(self.results.keys()),
            'RMSE': [self.results[k]['rmse'] for k in self.results.keys()],
            'MAE': [self.results[k]['mae'] for k in self.results.keys()],
            'R²': [self.results[k]['r2'] for k in self.results.keys()]
        })
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RMSE
        results_df.sort_values('RMSE').plot(x='Architecture', y='RMSE', 
                                            kind='bar', ax=axes[0], color='steelblue')
        axes[0].set_title('RMSE Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('RMSE')
        axes[0].set_xlabel('')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE
        results_df.sort_values('MAE').plot(x='Architecture', y='MAE', 
                                           kind='bar', ax=axes[1], color='coral')
        axes[1].set_title('MAE Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE')
        axes[1].set_xlabel('')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R² Score
        results_df.sort_values('R²', ascending=False).plot(x='Architecture', y='R²', 
                                                            kind='bar', ax=axes[2], color='green')
        axes[2].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('R² Score')
        axes[2].set_xlabel('')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('architecture_performance_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved: architecture_performance_comparison.png")
        
    def plot_training_curves(self):
        """Plot training and validation curves for all architectures."""
        n_models = len(self.histories)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, history) in enumerate(self.histories.items()):
            axes[idx].plot(history.history['loss'], label='Train Loss', linewidth=2)
            axes[idx].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
            axes[idx].set_title(f'{name} - Training History', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: training_curves_comparison.png")
    
    def plot_model_complexity_vs_performance(self):
        """Plot trade-off between model complexity and performance."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        complexity_df = pd.DataFrame({
            'Architecture': list(self.results.keys()),
            'Parameters': [self.results[k]['params'] for k in self.results.keys()],
            'RMSE': [self.results[k]['rmse'] for k in self.results.keys()],
            'Training Time': [self.results[k]['training_time'] for k in self.results.keys()]
        })
        
        # Parameters vs RMSE
        scatter1 = axes[0].scatter(complexity_df['Parameters'], complexity_df['RMSE'], 
                                   s=200, alpha=0.6, c=range(len(complexity_df)), cmap='viridis')
        for idx, row in complexity_df.iterrows():
            axes[0].annotate(row['Architecture'], 
                            (row['Parameters'], row['RMSE']),
                            fontsize=9, ha='center')
        axes[0].set_xlabel('Number of Parameters', fontsize=11)
        axes[0].set_ylabel('RMSE', fontsize=11)
        axes[0].set_title('Model Complexity vs Performance', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Training Time vs RMSE
        scatter2 = axes[1].scatter(complexity_df['Training Time'], complexity_df['RMSE'], 
                                   s=200, alpha=0.6, c=range(len(complexity_df)), cmap='plasma')
        for idx, row in complexity_df.iterrows():
            axes[1].annotate(row['Architecture'], 
                            (row['Training Time'], row['RMSE']),
                            fontsize=9, ha='center')
        axes[1].set_xlabel('Training Time (seconds)', fontsize=11)
        axes[1].set_ylabel('RMSE', fontsize=11)
        axes[1].set_title('Training Time vs Performance', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('complexity_vs_performance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: complexity_vs_performance.png")
    
    def plot_prediction_analysis(self, architecture: str = None):
        """Plot actual vs predicted values for best performing architecture."""
        if architecture is None:
            # Use best performing architecture
            comparison_df = pd.DataFrame({
                'Architecture': list(self.results.keys()),
                'RMSE': [self.results[k]['rmse'] for k in self.results.keys()]
            })
            architecture = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Architecture']
        
        model = self.results[architecture]['model']
        y_pred = model.predict(self.X_test, verbose=0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(self.y_test, y_pred, alpha=0.5, s=20)
        axes[0].plot([self.y_test.min(), self.y_test.max()], 
                    [self.y_test.min(), self.y_test.max()], 
                    'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Values', fontsize=11)
        axes[0].set_ylabel('Predicted Values', fontsize=11)
        axes[0].set_title(f'{architecture} - Actual vs Predicted', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = self.y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values', fontsize=11)
        axes[1].set_ylabel('Residuals', fontsize=11)
        axes[1].set_title(f'{architecture} - Residual Plot', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'prediction_analysis_{architecture.replace(" ", "_").lower()}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"✓ Saved: prediction_analysis_{architecture.replace(' ', '_').lower()}.png")
    
    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        report = "\n" + "="*80 + "\n"
        report += "NEURAL NETWORK ARCHITECTURE COMPARISON REPORT\n"
        report += "TOPSIS + NN Hybrid Model\n"
        report += "="*80 + "\n\n"
        
        comparison_df = pd.DataFrame({
            'Architecture': list(self.results.keys()),
            'RMSE': [self.results[k]['rmse'] for k in self.results.keys()],
            'MAE': [self.results[k]['mae'] for k in self.results.keys()],
            'R²': [self.results[k]['r2'] for k in self.results.keys()],
            'Parameters': [self.results[k]['params'] for k in self.results.keys()],
            'Time (s)': [self.results[k]['training_time'] for k in self.results.keys()]
        }).sort_values('RMSE')
        
        report += "PERFORMANCE METRICS:\n"
        report += "-" * 80 + "\n"
        report += comparison_df.to_string(index=False)
        report += "\n\n"
        
        best_rmse = comparison_df.iloc[0]['Architecture']
        best_r2 = comparison_df.sort_values('R²', ascending=False).iloc[0]['Architecture']
        smallest = comparison_df.sort_values('Parameters').iloc[0]['Architecture']
        fastest = comparison_df.sort_values('Time (s)').iloc[0]['Architecture']
        
        report += "RECOMMENDATIONS:\n"
        report += "-" * 80 + "\n"
        report += f"✓ Best RMSE Performance: {best_rmse}\n"
        report += f"✓ Best R² Score: {best_r2}\n"
        report += f"✓ Most Efficient (Smallest): {smallest}\n"
        report += f"✓ Fastest Training: {fastest}\n"
        report += "\nCONCLUSION:\n"
        report += "-" * 80 + "\n"
        report += "Use these results to select the best architecture for your TOPSIS + NN\n"
        report += "hybrid model based on your specific requirements (accuracy, speed, or size).\n"
        report += "="*80 + "\n"
        
        return report


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("NEURAL NETWORK ARCHITECTURE COMPARISON FOR TOPSIS + NN HYBRID MODEL")
    print("="*80)
    
    # Generate synthetic data
    print("\n[1/4] Generating synthetic MCDM data...")
    data_gen = SyntheticDataGenerator(n_samples=2000, n_criteria=8, noise_level=0.1)
    X_scaled, y_scaled, scaler = data_gen.get_normalized_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    print(f"✓ Generated {len(X_train)} training and {len(X_test)} test samples")
    
    # Initialize comparison framework
    print("\n[2/4] Training and comparing architectures...")
    comparator = ArchitectureComparison(X_train, X_test, y_train, y_test)
    comparison_results = comparator.compare_all(epochs=100, batch_size=32)
    
    # Display results
    print("\n[3/4] Generating visualizations...")
    print("\nCOMPARISON RESULTS:")
    print("-" * 80)
    print(comparison_results.to_string(index=False))
    
    # Generate plots
    comparator.plot_performance_comparison()
    comparator.plot_training_curves()
    comparator.plot_model_complexity_vs_performance()
    comparator.plot_prediction_analysis()
    
    # Generate report
    print("\n[4/4] Generating summary report...")
    report = comparator.generate_summary_report()
    print(report)
    
    # Save report to file
    with open('architecture_comparison_report.txt', 'w') as f:
        f.write(report)
    print("✓ Saved: architecture_comparison_report.txt")
    
    # Save detailed results to CSV
    comparison_results.to_csv('architecture_comparison_results.csv', index=False)
    print("✓ Saved: architecture_comparison_results.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nGenerated Files:")
    print("  - architecture_performance_comparison.png")
    print("  - training_curves_comparison.png")
    print("  - complexity_vs_performance.png")
    print("  - prediction_analysis_*.png")
    print("  - architecture_comparison_report.txt")
    print("  - architecture_comparison_results.csv")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
