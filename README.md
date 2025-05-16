# Amazon Book Reviews Anomaly Detection System

## Overview
This project implements an advanced anomaly detection system for Amazon book reviews using Deep Belief Networks (DBN) and various machine learning techniques. It provides a complete pipeline for data processing, model training, anomaly detection, and visualization.

## Features

### Data Processing
- Efficient data loading with support for both full and limited datasets
- Automatic feature extraction and preprocessing
- Memory-efficient handling of large datasets
- Support for Google Colab integration

### Model Components
- Deep Belief Network (DBN) implementation with RBM layers
- Multiple anomaly detection models:
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor
  - Elliptic Envelope
- Model tuning and hyperparameter optimization
- Automatic checkpointing and model state management

### Visualization
- Interactive visualization of results
- Learning curves and performance metrics
- Anomaly distribution analysis
- Feature importance visualization

## Dataset Source
The dataset used in this project is sourced from Kaggle, specifically the 'mohamedbakhet/amazon-books-reviews' dataset. It contains comprehensive reviews of books available on Amazon.

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd ads
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage
```python
from data_loader import DataLoader
from anomaly_detector import AnomalyDetector
from visualizer import Visualizer

# Initialize components
data_loader = DataLoader()
anomaly_detector = AnomalyDetector()
visualizer = Visualizer()

# Load and prepare data
df = data_loader.load_data(max_records=250000)  # Use limited dataset for faster processing
features = data_loader.prepare_features(df)
X_train, X_test, df_train, df_test = data_loader.split_train_test(features, df)

# Train anomaly detection models
results = anomaly_detector.train_and_evaluate(X_train, X_test)

# Visualize results
visualizer.plot_results(results)
```

### Advanced Usage
For more advanced usage, including model tuning and custom configurations, refer to the examples in the `src` directory.

## Project Structure

- `src/`
  - `main.py`: Main execution script
  - `data_loader.py`: Data loading and preprocessing
  - `dbn.py`: Deep Belief Network implementation
  - `anomaly_detector.py`: Anomaly detection models
  - `model_tuner.py`: Model tuning and optimization
  - `visualizer.py`: Visualization utilities
  - `visualization.py`: Additional visualization tools

## Dependencies

- pandas
- numpy
- scikit-learn
- kagglehub
- joblib
- matplotlib
- seaborn
- memory_profiler (optional)

## Running the Project

1. Ensure all dependencies are installed
2. Set up your Kaggle API credentials if using the full dataset
3. Run the main script:
```bash
python src/main.py
```

## Output

The system generates various outputs:
- Trained models saved as pickle files
- Performance metrics and results in JSON format
- Visualization plots and charts
- Model checkpoints for resuming training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]