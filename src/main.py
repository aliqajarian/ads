from src.data_loader import DataLoader
import sys
import os

# Detect Colab environment and adjust Python path
if 'google.colab' in sys.modules:
    sys.path.append('/content/ads')  # Add project root directory to path

from src.dbn import DeepBeliefNetwork
from src.anomaly_detector import AnomalyDetector
from src.visualizer import Visualizer # Corrected import path
import numpy as np
import os
import joblib # For saving anomaly detector if not using its own save method directly for some reason

# --- Google Colab/Drive Integration --- 
def mount_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
        return "/content/drive/MyDrive/ColabNotebooks/ads_output" # Example path, adjust as needed
    except ImportError:
        print("Not running in Google Colab or google.colab.drive is not available. Using local paths.")
        return "./ads_output" # Local fallback path

DRIVE_OUTPUT_PATH = mount_drive()
DBN_MODEL_PATH = os.path.join(DRIVE_OUTPUT_PATH, "dbn_model.pkl")
RBM_CHECKPOINT_PATH_PREFIX = os.path.join(DRIVE_OUTPUT_PATH, "rbm_checkpoints")
ANOMALY_DETECTOR_MODEL_PATH_TEMPLATE = os.path.join(DRIVE_OUTPUT_PATH, "anomaly_detector_{model_type}.pkl")

# Ensure output directories exist
os.makedirs(DRIVE_OUTPUT_PATH, exist_ok=True)
os.makedirs(RBM_CHECKPOINT_PATH_PREFIX, exist_ok=True)

def main():
    # Initialize data loader
    data_loader = DataLoader() # DataLoader now takes no arguments
    
    # Load and prepare data
    print("Loading data...")
    df = data_loader.load_data()
    features = data_loader.prepare_features(df)

    # Split data for DBN training and validation
    print("Splitting data for DBN training/validation...")
    # We use a portion of the data for DBN validation, not for the final anomaly detection test set
    # The final anomaly detection will still use the full 'features' transformed by the trained DBN
    X_train_dbn, X_val_dbn, _, _ = data_loader.split_train_test(features, df, test_size=0.15, random_state=42) # 15% for RBM validation
    
    # Define layer-specific configurations for DBN
    # Example: [{'n_iter': 20, 'learning_rate': 0.01}, {'n_iter': 15, 'learning_rate': 0.005}]
    # Adjust these based on your dataset and desired DBN architecture
    layer_configs = [
        {'n_iter': 25, 'learning_rate': 0.01, 'batch_size': 16},
        {'n_iter': 20, 'learning_rate': 0.005, 'batch_size': 16}
    ]
    # Ensure hidden_layers_sizes matches the number of configs
    # Example architecture, adjust as needed. Max components should not exceed input_dim of the layer.
    hidden_layers_sizes = [min(X_train_dbn.shape[1] // 2, X_train_dbn.shape[1]), min(X_train_dbn.shape[1] // 4, X_train_dbn.shape[1] // 2)] 

    # Initialize and train DBN
    print("Training Deep Belief Network...")
    # Check if a DBN model already exists
    if os.path.exists(DBN_MODEL_PATH):
        print(f"Loading existing DBN model from {DBN_MODEL_PATH}...")
        dbn = DeepBeliefNetwork.load_model(DBN_MODEL_PATH)
        # Ensure checkpoint path is updated if running in a new environment
        dbn.checkpoint_path_prefix = RBM_CHECKPOINT_PATH_PREFIX 
    else:
        print("No existing DBN model found. Training a new one...")
        dbn = DeepBeliefNetwork(
        hidden_layers_sizes=hidden_layers_sizes, 
        layer_configs=layer_configs, 
        random_state=42, 
        verbose=1,
        checkpoint_path_prefix=RBM_CHECKPOINT_PATH_PREFIX
    )
    dbn.fit(X_train_dbn, X_val=X_val_dbn)
    dbn.save_model(DBN_MODEL_PATH) # Save the trained DBN model
    
    # Transform the full feature set using the trained DBN
    transformed_features = dbn.transform(features)
    
    # Define anomaly detection models to compare
    model_configs = [
        {'model_type': 'isolation_forest', 'contamination': 0.1, 'params': {}},
        {'model_type': 'lof', 'contamination': 0.1, 'params': {'lof_neighbors': 20}},
        {'model_type': 'one_class_svm', 'params': {'svm_nu': 0.1, 'svm_kernel': 'rbf', 'svm_gamma': 'scale'}},
        {'model_type': 'elliptic_envelope', 'params': {'ee_contamination': 0.1}},
        {'model_type': 'dbscan', 'params': {'dbscan_eps': 0.5, 'dbscan_min_samples': 5}} # DBSCAN doesn't use 'contamination' directly
    ]

    all_anomalies_results = {}

    for config in model_configs:
        print(f"\nRunning anomaly detection with {config['model_type']}...")
        detector_params = {'model_type': config['model_type'], **config.get('params', {})}
        # Pass contamination if the model uses it directly (e.g. IsolationForest, LOF)
        if 'contamination' in config:
            detector_params['contamination'] = config['contamination']
        
        detector_model_path = ANOMALY_DETECTOR_MODEL_PATH_TEMPLATE.format(model_type=config['model_type'])
        if os.path.exists(detector_model_path):
            print(f"Loading existing {config['model_type']} anomaly detector from {detector_model_path}...")
            detector = AnomalyDetector.load_model(detector_model_path)
        else:
            print(f"No existing {config['model_type']} model found. Training a new one...")
            detector = AnomalyDetector(**detector_params)
            detector.fit(transformed_features)
            detector.save_model(detector_model_path) # Save the trained anomaly detector
            
        anomalies = detector.detect_anomalies(transformed_features)
        all_anomalies_results[config['model_type']] = anomalies
        
        print(f"Results for {config['model_type']}:")
        print(f"  Anomalies detected: {sum(anomalies)}")
        print(f"  Anomaly percentage: {(sum(anomalies)/len(anomalies))*100:.2f}%")

    # Visualize results - original visualizations + new comparison plot (to be added to visualizer.py)
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    visualizer.plot_rating_distribution(df)
    visualizer.plot_review_length_vs_rating(df)
    
    # Plot anomaly distribution for the first model as an example, or choose one
    if 'isolation_forest' in all_anomalies_results:
        visualizer.plot_anomaly_distribution(df, all_anomalies_results['isolation_forest'])
    elif all_anomalies_results:
        first_model_key = list(all_anomalies_results.keys())[0]
        visualizer.plot_anomaly_distribution(df, all_anomalies_results[first_model_key])
        
    visualizer.plot_dbn_layer_scores(dbn.layer_scores_)
    visualizer.plot_model_comparison(all_anomalies_results) # New comparison plot
    
    # Print summary
    print(f"\nTotal reviews analyzed: {len(df)}")
    for model_type, anomalies_result in all_anomalies_results.items():
        print(f"Model: {model_type} - Anomalies detected: {sum(anomalies_result)}, Percentage: {(sum(anomalies_result)/len(anomalies_result))*100:.2f}%")

if __name__ == "__main__":
    main()