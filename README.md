# Hybrid TOPSIS + Neural Network Project

A sophisticated multi-criteria decision-making (MCDM) system that combines the TOPSIS (Technique for Order Preference by Similarity to Ideal Solution) algorithm with neural network capabilities for advanced decision analysis and prediction.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a hybrid approach combining:
- **TOPSIS Algorithm**: A classical multi-criteria decision-making technique for ranking alternatives based on their similarity to ideal and anti-ideal solutions
- **Neural Networks**: Deep learning models for pattern recognition, feature extraction, and predictive analytics
- **Data Preprocessing**: Robust data normalization, weighting, and feature engineering

The hybrid system leverages TOPSIS's interpretability with neural networks' predictive power to solve complex decision-making problems.

## ✨ Features

- **Multi-Criteria Analysis**: Evaluate alternatives against multiple criteria simultaneously
- **TOPSIS Implementation**: Full TOPSIS algorithm with weighted criteria support
- **Neural Network Integration**: Deep learning models for enhanced decision-making
- **Data Preprocessing**: Automatic normalization and feature scaling
- **Visualization**: Comprehensive charts and graphs for decision analysis
- **Model Persistence**: Save and load trained models
- **Configurable Parameters**: Flexible configuration for different use cases
- **Performance Metrics**: Detailed evaluation and validation metrics

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/vkas0812/vkas0812.git
cd vkas0812
git checkout mcdm_project
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Dependencies

The project requires the following packages:
- **numpy**: Numerical computing and array operations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms and preprocessing
- **tensorflow** or **pytorch**: Neural network framework
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scipy**: Scientific computing

Install them manually if needed:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn scipy
```

## 📁 Project Structure

```
vkas0812/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup configuration
│
├── src/
│   ├── __init__.py
│   ├── main.py                       # Main entry point
│   │
│   ├── topsis/
│   │   ├── __init__.py
│   │   ├── topsis_core.py           # Core TOPSIS algorithm
│   │   ├── normalization.py         # Data normalization methods
│   │   └── weighting.py             # Criteria weighting
│   │
│   ├── neural_network/
│   │   ├── __init__.py
│   │   ├── models.py                # Neural network models
│   │   ├── trainer.py               # Model training pipeline
│   │   └── predictor.py             # Prediction module
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Data input/output
│   │   ├── feature_engineering.py   # Feature creation
│   │   └── validation.py            # Data validation
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── plots.py                 # Plotting functions
│   │   └── dashboards.py            # Dashboard creation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       ├── logger.py                # Logging utilities
│       └── metrics.py               # Evaluation metrics
│
├── data/
│   ├── raw/                         # Raw input data
│   ├── processed/                   # Processed data
│   └── sample_data.csv              # Sample dataset
│
├── models/
│   ├── topsis_weights.pkl           # Saved TOPSIS weights
│   └── trained_model.h5             # Saved neural network
│
├── results/
│   ├── rankings.csv                 # TOPSIS rankings
│   ├── predictions.csv              # Neural network predictions
│   └── visualizations/              # Generated plots
│
├── tests/
│   ├── __init__.py
│   ├── test_topsis.py              # TOPSIS tests
│   ├── test_neural_network.py      # NN tests
│   └── test_preprocessing.py       # Preprocessing tests
│
└── notebooks/
    ├── topsis_analysis.ipynb        # TOPSIS analysis notebook
    ├── model_training.ipynb         # Model training notebook
    └── results_visualization.ipynb  # Results visualization
```

## 💡 Usage Examples

### Basic TOPSIS Analysis

```python
from src.topsis.topsis_core import TOPSIS
from src.preprocessing.data_loader import DataLoader
import pandas as pd

# Load data
loader = DataLoader()
data = loader.load_csv('data/raw/alternatives.csv')

# Initialize TOPSIS with criteria and weights
criteria = ['cost', 'quality', 'delivery', 'reliability']
weights = [0.3, 0.3, 0.2, 0.2]
impacts = ['negative', 'positive', 'positive', 'positive']  # negative for cost

# Perform TOPSIS analysis
topsis = TOPSIS(data, criteria, weights, impacts)
rankings = topsis.calculate()

# Display results
print(rankings)
rankings.to_csv('results/topsis_rankings.csv', index=False)
```

### Neural Network Model Training

```python
from src.neural_network.models import HybridModel
from src.neural_network.trainer import ModelTrainer
from src.preprocessing.data_loader import DataLoader

# Load and preprocess data
loader = DataLoader()
X_train, y_train = loader.load_training_data('data/processed/train.csv')
X_test, y_test = loader.load_testing_data('data/processed/test.csv')

# Create and train model
model = HybridModel(input_size=X_train.shape[1])
trainer = ModelTrainer(model)

history = trainer.train(
    X_train, y_train,
    X_test, y_test,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

# Save trained model
trainer.save_model('models/trained_model.h5')
```

### Hybrid Analysis (TOPSIS + Neural Network)

```python
from src.topsis.topsis_core import TOPSIS
from src.neural_network.predictor import Predictor
import pandas as pd

# Step 1: Perform TOPSIS for initial ranking
topsis_rankings = topsis.calculate()

# Step 2: Use neural network for confidence scoring
predictor = Predictor('models/trained_model.h5')
topsis_rankings['confidence_score'] = predictor.predict(topsis_rankings[features])

# Step 3: Combined ranking
topsis_rankings['hybrid_rank'] = (
    0.6 * topsis_rankings['topsis_score'] + 
    0.4 * topsis_rankings['confidence_score']
).rank(ascending=False)

print(topsis_rankings.sort_values('hybrid_rank'))
```

### Data Visualization

```python
from src.visualization.plots import plot_rankings, plot_criteria_comparison
from src.visualization.dashboards import create_analysis_dashboard

# Plot TOPSIS rankings
plot_rankings(rankings, save_path='results/visualizations/rankings.png')

# Compare criteria importance
plot_criteria_comparison(weights, criteria, save_path='results/visualizations/criteria.png')

# Create comprehensive dashboard
create_analysis_dashboard(rankings, results, save_path='results/visualizations/dashboard.html')
```

## ⚙️ Configuration

Create a `config.yaml` file in the project root:

```yaml
# Data Configuration
data:
  raw_path: 'data/raw'
  processed_path: 'data/processed'
  sample_size: 1000

# TOPSIS Configuration
topsis:
  normalization_method: 'vector'  # or 'linear'
  epsilon: 1e-10
  
# Neural Network Configuration
neural_network:
  framework: 'tensorflow'  # or 'pytorch'
  architecture:
    layers: [128, 64, 32, 16]
    activation: 'relu'
    dropout: 0.3
  training:
    epochs: 100
    batch_size: 32
    learning_rate: 0.001
    optimizer: 'adam'

# Logging Configuration
logging:
  level: 'INFO'
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

## 📚 API Reference

### TOPSIS Core Module

```python
TOPSIS(data, criteria, weights, impacts)
    .calculate()                    # Returns DataFrame with rankings
    .get_ideal_solution()          # Returns ideal solution
    .get_anti_ideal_solution()     # Returns anti-ideal solution
    .get_distances()               # Returns separation measures
```

### Neural Network Module

```python
HybridModel(input_size, hidden_layers=[128, 64, 32])
    .build()                        # Builds the model
    .summary()                      # Prints model summary

ModelTrainer(model)
    .train(X_train, y_train, ...)  # Trains the model
    .evaluate(X_test, y_test)      # Evaluates on test data
    .save_model(path)              # Saves trained model
```

### Data Preprocessing Module

```python
DataLoader()
    .load_csv(path)                 # Loads CSV file
    .load_training_data(path)      # Loads training dataset
    .preprocess(data)              # Applies preprocessing pipeline
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_topsis.py

# Run with coverage report
pytest --cov=src tests/
```

## 📊 Example Results

The project generates various outputs:

- **rankings.csv**: TOPSIS rankings with scores and distances
- **predictions.csv**: Neural network predictions
- **visualizations/**: PNG plots and interactive dashboards
- **models/**: Saved trained models for future use

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Create a new branch for your feature: `git checkout -b feature/your-feature`
2. Make your changes and commit: `git commit -m 'Add your feature'`
3. Push to your branch: `git push origin feature/your-feature`
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Vikrant Singh (vkas0812)**

For questions or support, please open an issue on the GitHub repository.

## 🔗 References

- Hwang, C. L., & Yoon, K. (1981). Multiple Attribute Decision Making: Methods and Applications.
- Behzadian, M., et al. (2012). A state-of the art survey of TOPSIS applications. Expert Systems with Applications.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.

---

**Last Updated**: December 23, 2025

For the latest version and updates, visit the [GitHub Repository](https://github.com/vkas0812/vkas0812)
