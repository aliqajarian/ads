# Amazon Book Reviews Anomaly Detection

This project implements an advanced anomaly detection system for Amazon book reviews using a combination of Deep Belief Networks (DBN) and multiple anomaly detection algorithms. The system is designed to identify suspicious or anomalous reviews based on various behavioral and textual features.

## Dataset

This project uses the Amazon Books Reviews dataset from Kaggle. The dataset contains comprehensive book reviews from Amazon, which serves as the foundation for our anomaly detection system.

**Dataset Source**: [Amazon Books Reviews on Kaggle](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews)

## Features

- **Deep Belief Network (DBN) Feature Extraction**
  - Multi-layer RBM architecture
  - Automatic feature dimensionality reduction
  - Layer-wise training with checkpointing
  - Pseudo-likelihood monitoring

- **Multiple Anomaly Detection Models**
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - Histogram-based Outlier Score (HBOS)
  - DBSCAN

- **Comprehensive Model Tuning**
  - Automated hyperparameter optimization
  - Learning curve analysis
  - Cross-validation performance metrics
  - Results saved in both JSON and CSV formats

- **Advanced Visualization**
  - t-SNE visualization of DBN features
  - Behavioral features correlation heatmap
  - Learning curves for model sensitivity analysis
  - Rating distribution analysis
  - Review length vs. rating analysis
  - Anomaly distribution visualization
  - Model comparison plots

## Project Structure

```
ads/
├── src/
│   ├── __init__.py
│   ├── main.py                 # Main execution script
│   ├── data_downloader.py      # Data downloading utilities
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── dbn.py                  # Deep Belief Network implementation
│   ├── model_tuner.py          # Model tuning and evaluation
│   ├── visualizer.py           # Visualization utilities
│   └── anomaly_detector.py     # Anomaly detection models
├── data/
│   └── raw/                    # Raw data directory
├── ads_output/                 # Output directory for results
│   ├── tsne_visualizations/    # t-SNE plots
│   ├── correlation_visualizations/  # Correlation heatmaps
│   ├── learning_curves/        # Learning curve plots
│   └── model_tuning_results/   # Tuning results and metrics
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ads.git
cd ads
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python src/main.py
```

The script will:
- Download and preprocess the Amazon book reviews dataset
- Train the Deep Belief Network
- Perform hyperparameter tuning for all anomaly detection models
- Generate comprehensive visualizations
- Save results and metrics

## Output

The system generates several types of outputs:

1. **Model Results**
   - JSON file with detailed model metrics
   - CSV files with tuning summaries
   - Learning curve analysis results

2. **Visualizations**
   - t-SNE plots for each model
   - Behavioral features correlation heatmap
   - Learning curves for model sensitivity
   - Various distribution and comparison plots

3. **Saved Models**
   - DBN model checkpoints
   - Trained anomaly detection models

## Model Tuning

The system automatically tunes the following parameters for each model:

- **Isolation Forest**
  - Number of estimators
  - Contamination rate
  - Maximum samples

- **Local Outlier Factor**
  - Number of neighbors
  - Contamination rate
  - Distance metric

- **One-Class SVM**
  - Nu parameter
  - Kernel type
  - Gamma parameter

- **HBOS**
  - Number of bins
  - Alpha parameter
  - Contamination rate

- **DBSCAN**
  - Epsilon parameter
  - Minimum samples
  - Distance metric

## Visualization Features

The visualization module provides:

1. **t-SNE Visualization**
   - 2D projection of DBN features
   - Color-coded anomaly detection
   - Interactive legend

2. **Correlation Analysis**
   - Behavioral features correlation heatmap
   - Annotated correlation values
   - Custom color scheme

3. **Learning Curves**
   - Training and validation scores
   - Standard deviation bands
   - Model comparison

4. **Distribution Analysis**
   - Rating distribution
   - Review length analysis
   - Anomaly distribution

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.