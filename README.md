# Amazon Book Reviews Dataset Processor

## Overview
This project processes Amazon book reviews data for anomaly detection and analysis. It includes functionality to download, preprocess, and extract features from the dataset.

## Recent Updates

### Limited Dataset Support
The `DataLoader` class now supports loading a limited subset of the data (default: 500k records) for faster processing during development and testing. This is particularly useful when working with the large Amazon book reviews dataset.

## Usage

### Loading Limited Dataset (500k records)
```python
from data_loader import DataLoader

# Initialize the data loader
data_loader = DataLoader()

# Load limited dataset (default: 500k records)
df_limited = data_loader.load_data(max_records=500000, use_full_dataset=False)

# Prepare features
features_limited = data_loader.prepare_features(df_limited)

# Split into train and test sets
X_train, X_test, df_train, df_test = data_loader.split_train_test(features_limited, df_limited)
```

### Loading Full Dataset
```python
from data_loader import DataLoader

# Initialize the data loader
data_loader = DataLoader()

# Load the full dataset
df_full = data_loader.load_data(use_full_dataset=True)

# Prepare features
features_full = data_loader.prepare_features(df_full)

# Split into train and test sets
X_train, X_test, df_train, df_test = data_loader.split_train_test(features_full, df_full)
```

## Implementation Details

The limited dataset functionality is implemented in the `load_data` method of the `DataLoader` class using stratified sampling based on review scores to maintain the distribution of the original dataset.