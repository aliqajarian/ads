from data_loader import DataLoader
import sys
import os
import gc
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from dbn import DeepBeliefNetwork
from anomaly_detector import AnomalyDetector
from visualizer import Visualizer
from model_tuner import ModelTuner
import numpy as np
import os
import joblib
import warnings
from contextlib import contextmanager


# Detect Colab environment and adjust Python path
if 'google.colab' in sys.modules:
    sys.path.append('/content/ads')
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully.")
        DRIVE_OUTPUT_PATH = "/content/drive/MyDrive/ads/ads_output"
    except Exception as e:
        print(f"Error mounting Google Drive: {str(e)}")
        DRIVE_OUTPUT_PATH = "/root/code/ads_output"
else:
    print("Not running in Google Colab. Using local paths.")
    #DRIVE_OUTPUT_PATH = "/content/drive/MyDrive/ads/ads_output"
    DRIVE_OUTPUT_PATH = "/root/code/ads_output"

# Define paths
DBN_MODEL_PATH = os.path.join(DRIVE_OUTPUT_PATH, "dbn_model.pkl")
RBM_CHECKPOINT_PATH_PREFIX = os.path.join(DRIVE_OUTPUT_PATH, "rbm_checkpoints")
ANOMALY_DETECTOR_MODEL_PATH_TEMPLATE = os.path.join(DRIVE_OUTPUT_PATH, "anomaly_detector_{model_type}.pkl")
RESULTS_PATH = os.path.join(DRIVE_OUTPUT_PATH, "model_results.json")

# Ensure output directories exist
os.makedirs(DRIVE_OUTPUT_PATH, exist_ok=True)
os.makedirs(RBM_CHECKPOINT_PATH_PREFIX, exist_ok=True)

def save_results(results, filepath):
    """Save model results to a JSON file."""
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
            for k, v in metrics.items()
        }
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Results saved to {filepath}")

@contextmanager
def memory_cleanup():
    """Context manager for cleaning up memory after operations."""
    try:
        yield
    finally:
        gc.collect()
        if 'google.colab' in sys.modules:
            try:
                from google.colab import runtime
                runtime.unassign()
            except:
                pass

def handle_large_data(func):
    """Decorator to handle large data operations safely."""
    def wrapper(*args, **kwargs):
        with memory_cleanup():
            try:
                return func(*args, **kwargs)
            except MemoryError:
                print("Memory error encountered. Try reducing the dataset size or batch size.")
                raise
            except Exception as e:
                print(f"Error in {func.__name__}: {str(e)}")
                raise
    return wrapper

@handle_large_data
def train_dbn(X_train, X_val, hidden_layers_sizes, layer_configs):
    """Train or load DBN model with optimized parameters for faster training."""
    try_load_model = False
    if os.path.exists(DBN_MODEL_PATH):
        try:
            print(f"Loading existing DBN model from {DBN_MODEL_PATH}...")
            dbn = DeepBeliefNetwork.load_model(DBN_MODEL_PATH)
            dbn.checkpoint_path_prefix = RBM_CHECKPOINT_PATH_PREFIX
            try_load_model = True
        except Exception as e:
            print(f"Error loading existing model, will train new one: {str(e)}")
    
    if not try_load_model:
        print("Training new DBN model with optimized parameters...")
        dbn = DeepBeliefNetwork(
            hidden_layers_sizes=hidden_layers_sizes,
            layer_configs=layer_configs,
            random_state=42,
            verbose=1,
            checkpoint_path_prefix=RBM_CHECKPOINT_PATH_PREFIX,
            early_stopping=True,
            patience=2,  # Reduced patience for faster convergence
            n_jobs=-1,   # Use all available cores
            use_tqdm=True,
            feature_subset_ratio=0.8  # Use 80% of features for faster training
        )
        try:
            dbn.fit(X_train, X_val=X_val)
            dbn.save_model(DBN_MODEL_PATH)
        except Exception as e:
            print(f"Error during model training/saving: {str(e)}")
            raise
    return dbn

try:
    from memory_profiler import profile
    MEMORY_PROFILING_ENABLED = True
except ImportError:
    print("Memory profiling disabled - memory_profiler package not installed")
    MEMORY_PROFILING_ENABLED = False
    def profile(func):
        return func

@profile
def main():
    # Initialize data loader
    data_loader = DataLoader()
    
    # Configure warnings and memory settings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Load and prepare data with dynamic batch sizing
    print("Loading data with optimized batch size...")
    try:
        max_records = 100000 if 'google.colab' in sys.modules else 250000
        df = data_loader.load_data(max_records=max_records, use_full_dataset=False)
    except MemoryError:
        print("Memory limit reached. Reducing dataset size...")
        df = data_loader.load_data(max_records=50000, use_full_dataset=False)
    features = data_loader.prepare_features(df)

    # Split data for DBN training and validation
    print("Splitting data for DBN training/validation...")
    X_train_dbn, X_val_dbn, _, _ = data_loader.split_train_test(features, df, test_size=0.15, random_state=42)
    
    # Define layer-specific configurations for DBN with optimized parameters for faster training
    layer_configs = [
        {'n_iter': 10, 'learning_rate': 0.05, 'batch_size': 256},
        {'n_iter': 8, 'learning_rate': 0.03, 'batch_size': 256}
    ]
    
    hidden_layers_sizes = [min(X_train_dbn.shape[1] // 2, X_train_dbn.shape[1]), 
                          min(X_train_dbn.shape[1] // 4, X_train_dbn.shape[1] // 2)]

    # Train DBN with optimized parameters and memory management
    try:
        with memory_cleanup():
            dbn = train_dbn(X_train_dbn, X_val_dbn, hidden_layers_sizes, layer_configs)
    except Exception as e:
        print(f"Error during DBN training: {str(e)}")
        raise
    
    # Transform features with memory optimization
    try:
        with memory_cleanup():
            transformed_features = dbn.transform(features)
            del features  # Free up memory
            gc.collect()
    except Exception as e:
        print(f"Error during feature transformation: {str(e)}")
        raise
    
    # Assuming we have some ground truth labels
    y_true = np.zeros(len(transformed_features))
    y_true[:int(len(y_true) * 0.1)] = 1  # Assuming 10% are anomalies
    
    # Initialize model tuner with memory management
    try:
        with memory_cleanup():
            model_tuner = ModelTuner(DRIVE_OUTPUT_PATH)
            
            # Perform hyperparameter tuning with progress tracking
            print("\nStarting hyperparameter tuning with memory optimization...")
            best_params, tuning_results = model_tuner.tune_models(transformed_features, y_true)
            
            # Clean up memory before learning curve analysis
            gc.collect()
            
            print("\nAnalyzing learning curves with memory optimization...")
            learning_curve_results = model_tuner.analyze_learning_curves(transformed_features, y_true)
    except Exception as e:
        print(f"Error during model tuning or learning curve analysis: {str(e)}")
        raise
    
    # Update model configs with best parameters
    model_configs = [
        {
            'model_type': 'isolation_forest',
            'contamination': best_params['isolation_forest']['contamination'],
            'params': {
                'n_estimators': best_params['isolation_forest']['n_estimators'],
                'max_samples': best_params['isolation_forest']['max_samples']
            }
        },
        {
            'model_type': 'lof',
            'contamination': best_params['lof']['contamination'],
            'params': {
                'lof_neighbors': best_params['lof']['n_neighbors'],
                'metric': best_params['lof']['metric']
            }
        },
        {
            'model_type': 'one_class_svm',
            'params': {
                'svm_nu': best_params['one_class_svm']['nu'],
                'svm_kernel': best_params['one_class_svm']['kernel'],
                'svm_gamma': best_params['one_class_svm']['gamma']
            }
        },
        {
            'model_type': 'hbos',
            'contamination': best_params['hbos']['contamination'],
            'params': {
                'hbos_n_bins': best_params['hbos']['n_bins'],
                'hbos_alpha': best_params['hbos']['alpha']
            }
        },
        {
            'model_type': 'birch',
            'params': {
                'threshold': best_params['birch']['threshold'],
                'branching_factor': best_params['birch']['branching_factor'],
                'n_clusters': best_params['birch']['n_clusters'],
                'compute_labels': best_params['birch']['compute_labels']
            }
        }
    ]

    # Initialize results containers and manage memory during model evaluation
    all_anomalies_results = {}
    model_metrics = {}
    
    try:
        with memory_cleanup():
            for config in model_configs:
                print(f"\nRunning anomaly detection with {config['model_type']}...")
                detector_params = {'model_type': config['model_type'], **config.get('params', {})}
                if 'contamination' in config:
                    detector_params['contamination'] = config['contamination']
                
                detector_model_path = ANOMALY_DETECTOR_MODEL_PATH_TEMPLATE.format(model_type=config['model_type'])
                try_load_model = False
                if os.path.exists(detector_model_path):
                    try:
                        print(f"Loading existing {config['model_type']} anomaly detector from {detector_model_path}...")
                        detector = AnomalyDetector.load_model(detector_model_path)
                        try_load_model = True
                    except (OSError, EOFError, ValueError) as e:
                        print(f"Error loading existing {config['model_type']} model (corrupted file), will train new one: {str(e)}")
                    except Exception as e:
                        print(f"Unexpected error loading {config['model_type']} model: {str(e)}")
                
                if not try_load_model:
                    print(f"Training new {config['model_type']} model...")
                    detector = AnomalyDetector(**detector_params)
                    try:
                        detector.fit(transformed_features)
                        detector.save_model(detector_model_path)
                    except MemoryError:
                        print(f"Memory error during {config['model_type']} model training. Trying with reduced feature set...")
                        # Try training with reduced feature set
                        feature_subset = transformed_features[:, :transformed_features.shape[1]//2]
                        detector.fit(feature_subset)
                        detector.save_model(detector_model_path)
                    except Exception as e:
                        print(f"Error training/saving {config['model_type']} model: {str(e)}")
                        raise
                    
                anomalies = detector.detect_anomalies(transformed_features)
                all_anomalies_results[config['model_type']] = anomalies
                
                # Calculate metrics
                precision = precision_score(y_true, anomalies)
                recall = recall_score(y_true, anomalies)
                f1 = f1_score(y_true, anomalies)
                
                # Store metrics
                model_metrics[config['model_type']] = {
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1,
                    "Anomalies_Detected": int(sum(anomalies)),
                    "Anomaly_Percentage": float((sum(anomalies)/len(anomalies))*100)
                }
        
        print(f"Results for {config['model_type']}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        
        # Clean up memory after each model evaluation
        gc.collect()
    except Exception as e:
        print(f"Error during anomaly detection: {str(e)}")
        raise
    finally:
        # Final memory cleanup
        gc.collect()

    # Save results to file
    save_results(model_metrics, RESULTS_PATH)

    # Visualize results
    print("\nGenerating visualizations...")
    visualizer = Visualizer()
    
    # Create output directories for visualizations
    tsne_output_path = os.path.join(DRIVE_OUTPUT_PATH, "tsne_visualizations")
    correlation_output_path = os.path.join(DRIVE_OUTPUT_PATH, "correlation_visualizations")
    learning_curves_path = os.path.join(DRIVE_OUTPUT_PATH, "learning_curves")
    os.makedirs(tsne_output_path, exist_ok=True)
    os.makedirs(correlation_output_path, exist_ok=True)
    os.makedirs(learning_curves_path, exist_ok=True)
    
    # Plot behavioral features correlation
    print("\nGenerating behavioral features correlation heatmap...")
    correlation_save_path = os.path.join(correlation_output_path, "behavioral_features_correlation.png")
    visualizer.plot_behavioral_features_correlation(df, save_path=correlation_save_path)
    
    # Plot learning curves for model sensitivity analysis
    print("\nGenerating learning curves for model sensitivity analysis...")
    learning_curves_save_path = os.path.join(learning_curves_path, "model_learning_curves.png")
    visualizer.plot_learning_curves(transformed_features, y_true, save_path=learning_curves_save_path)
    
    # Plot other visualizations
    visualizer.plot_rating_distribution(df)
    visualizer.plot_review_length_vs_rating(df)
    
    # Plot t-SNE visualization for each model's results
    for model_type, anomalies in all_anomalies_results.items():
        print(f"\nGenerating t-SNE visualization for {model_type}...")
        tsne_save_path = os.path.join(tsne_output_path, f"tsne_{model_type}.png")
        visualizer.plot_tsne_features(transformed_features, anomalies, save_path=tsne_save_path)
    
    # Plot anomaly distribution for the first model as an example
    if 'isolation_forest' in all_anomalies_results:
        visualizer.plot_anomaly_distribution(df, all_anomalies_results['isolation_forest'])
    elif all_anomalies_results:
        first_model_key = list(all_anomalies_results.keys())[0]
        visualizer.plot_anomaly_distribution(df, all_anomalies_results[first_model_key])
        
    visualizer.plot_dbn_layer_scores(dbn.layer_scores_)
    visualizer.plot_model_comparison(all_anomalies_results)
    
    # Print summary
    print(f"\nTotal reviews analyzed: {len(df)}")
    for model_type, metrics in model_metrics.items():
        print(f"\nModel: {model_type}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1 Score: {metrics['F1']:.4f}")
        print(f"  Anomalies detected: {metrics['Anomalies_Detected']}")
        print(f"  Anomaly percentage: {metrics['Anomaly_Percentage']:.2f}%")

if __name__ == "__main__":
    main()