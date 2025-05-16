# Amazon Book Reviews Anomaly Detection System

A comprehensive system for detecting anomalies in Amazon book reviews using various machine learning models and visualization techniques.

## Overview

This project implements an anomaly detection system for Amazon book review data, utilizing multiple models including Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, and HBOS. The system provides extensive visualization capabilities to analyze and compare model performance.

## Dataset

The project uses the Amazon Book Reviews dataset from the McAuley Lab's Amazon Reviews 2023 collection <mcreference link="https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023" index="1">1</mcreference>. The dataset includes:

- 29.5M book reviews
- 10.3M unique users
- 4.4M unique items
- 2.9B review text tokens
- Features include ratings, review text, helpfulness votes, and timestamps
- Data spans from May 1996 to September 2023

## Features

### Anomaly Detection
- Multiple anomaly detection models:
  - Isolation Forest
  - Local Outlier Factor (LOF)
  - One-Class SVM
  - Histogram-based Outlier Score (HBOS)
- Model comparison and evaluation
- Configurable anomaly thresholds

### Visualization
- Interactive visualization of results
- t-SNE feature space visualization
- Behavioral features correlation analysis
- Rating distribution analysis
- Review length vs. rating analysis
- Anomaly distribution visualization
- DBN layer scores visualization
- Model comparison plots
- ROC curves comparison

## Project Structure

```
src/
├── main.py              # Main execution script
├── data_loader.py       # Data loading and preprocessing
├── dbn.py              # Deep Belief Network implementation
├── anomaly_detector.py  # Anomaly detection models
├── model_tuner.py      # Model tuning and optimization
├── visualizer.py       # Visualization utilities
└── vis_mod.py          # Enhanced visualization module
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

1. Prepare your data:
```python
from src.data_loader import DataLoader

data_loader = DataLoader()
df = data_loader.load_data()
```

2. Run anomaly detection:
```python
from src.anomaly_detector import AnomalyDetector

detector = AnomalyDetector()
anomalies = detector.detect(df)
```

3. Generate visualizations:
```python
from src.vis_mod import generate_visualizations

generate_visualizations(df, features, anomalies, model_results, layer_scores)
```

## Output

The system generates various visualization outputs in the `ads_output` directory:

```
ads_output/
├── tsne_visualizations/
├── correlation_visualizations/
├── learning_curves/
└── plots/
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- pyod

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.