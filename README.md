# Anomaly Detection System (ADS)

This project implements an end-to-end anomaly detection pipeline using Deep Belief Networks (DBN) for feature extraction and multiple anomaly detection algorithms. It is designed for analyzing review datasets and identifying anomalous patterns, with built-in support for Google Colab and Google Drive integration for easy experimentation and model checkpointing.

## Features
- **Deep Belief Network (DBN)**: Layer-wise unsupervised feature extraction using BernoulliRBM layers.
- **Multiple Anomaly Detection Models**: Isolation Forest, Local Outlier Factor (LOF), One-Class SVM, Elliptic Envelope, and DBSCAN.
- **Visualization**: Plots for rating distribution, review length vs. rating, anomaly distributions, DBN layer scores, and model comparison.
- **Google Colab/Drive Integration**: Seamless saving/loading of models and checkpoints to Google Drive.
- **Modular Codebase**: Easily extensible for new datasets or anomaly detection methods.

## Project Structure
```
ads/
├── src/
│   ├── main.py              # Main pipeline script
│   ├── dbn.py               # Deep Belief Network implementation
│   ├── anomaly_detector.py  # Anomaly detection models
│   ├── data_loader.py       # Data loading and preprocessing
│   ├── visualizer.py        # Visualization utilities
│   └── ...
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Requirements
- Python 3.7+
- numpy, pandas, scikit-learn, joblib, matplotlib
- Google Colab (for cloud execution)

Install dependencies locally:
```bash
pip install -r requirements.txt
```

## Getting Started

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/aliqajarian/ads.git
cd ads
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python src/main.py
```

All outputs and model checkpoints will be saved in the `ads_output` directory by default.

### Running on Google Colab

1. Clone the repository to your Google Drive:
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone the repository
!git clone https://github.com/aliqajarian/ads.git /content/ads
%cd /content/ads
```

2. Install dependencies:
```python
!pip install -r requirements.txt
```

3. Run the main script:
```python
!python src/main.py
```

All model checkpoints and outputs will be saved to your Google Drive under `ads_output` (as configured in `main.py`).

**Note**: Replace `aliqajarian` in the Git URLs with your actual GitHub username.

## Dataset
- Place your dataset in the appropriate location as expected by `data_loader.py` (see its docstring or code for details).
- The default pipeline expects a DataFrame with review data; adjust `data_loader.py` as needed for your dataset.

## Customization
- **DBN Architecture**: Change `hidden_layers_sizes` and `layer_configs` in `main.py` to experiment with different DBN structures.
- **Anomaly Detectors**: Add or modify model configs in `main.py` to try different algorithms or parameters.
- **Visualizations**: Extend `visualizer.py` for more plots or analytics.

## Interpreting Results
- The script prints the number and percentage of anomalies detected by each model.
- Visualizations are generated for data distribution, anomaly detection results, and model comparisons.
- Checkpoints and trained models are saved for reproducibility and further analysis.

## Troubleshooting
- If running outside Colab, the script will default to local paths for outputs.
- Ensure all dependencies are installed and dataset paths are correct.

## License
MIT License