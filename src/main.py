from data_loader import DataLoader
import sys
import os
import gc
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json
from dbn import DeepBeliefNetwork
from anomaly_detector import AnomalyDetector
from visualizer import Visualizer
from model_tuner import ModelTuner
import numpy as np
import os
import joblib
import warnings
from datetime import datetime
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

# Define JSON checkpoint paths
TUNING_CHECKPOINT_PATH = os.path.join(DRIVE_OUTPUT_PATH, "tuning_checkpoint.json")
LEARNING_CURVES_CHECKPOINT_PATH = os.path.join(DRIVE_OUTPUT_PATH, "learning_curves_checkpoint.json")
MODEL_STATE_PATH = os.path.join(DRIVE_OUTPUT_PATH, "model_state.json")

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

def load_model_results(directory, model_type=None):
    """Load saved model results from JSON files.
    
    Args:
        directory: Directory containing the saved model results
        model_type: Optional specific model type to load. If None, loads all models.
        
    Returns:
        dict: Dictionary of model results
    """
    results = {}
    
    if model_type:
        # Load specific model results
        model_path = os.path.join(directory, f"anomaly_results_{model_type}.json")
        if os.path.exists(model_path):
            try:
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                    results[model_type] = {
                        'anomalies': np.array(model_data['anomalies']),
                        'metrics': model_data['metrics'],
                        'params': model_data.get('params', {}),
                        'timestamp': model_data.get('timestamp')
                    }
                print(f"Loaded results for {model_type} from {model_path}")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading results for {model_type}: {e}")
    else:
        # Load all model results
        result_files = [f for f in os.listdir(directory) if f.startswith("anomaly_results_") and f.endswith(".json")]
        for file in result_files:
            model_type = file.replace("anomaly_results_", "").replace(".json", "")
            model_path = os.path.join(directory, file)
            try:
                with open(model_path, 'r') as f:
                    model_data = json.load(f)
                    results[model_type] = {
                        'anomalies': np.array(model_data['anomalies']),
                        'metrics': model_data['metrics'],
                        'params': model_data.get('params', {}),
                        'timestamp': model_data.get('timestamp')
                    }
                print(f"Loaded results for {model_type} from {model_path}")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading results for {model_type}: {e}")
    
    return results

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

def check_model_state(state_path):
    """Check if a model state file exists and load it.
    
    Args:
        state_path: Path to the model state file
        
    Returns:
        dict: Model state if exists, None otherwise
    """
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                print(f"Loaded existing model state from {state_path}")
                print(f"Status: {state.get('status', 'unknown')}")
                print(f"Last updated: {state.get('last_updated', state.get('timestamp', 'unknown'))}")
                return state
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading model state: {e}")
    return None

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
            
            # Check for existing model state to resume from
            existing_model_state = check_model_state(MODEL_STATE_PATH)
            
            # Initialize or load model state
            if existing_model_state and existing_model_state.get('status') in ['tuning_completed', 'learning_curves_completed']:
                print("\nResuming from previous model state...")
                model_state = existing_model_state
                best_params = model_state.get('best_params', {})
                
                # Check if we need to load tuning results from checkpoint
                if not best_params and os.path.exists(TUNING_CHECKPOINT_PATH):
                    print(f"Loading tuning results from {TUNING_CHECKPOINT_PATH}")
                    try:
                        with open(TUNING_CHECKPOINT_PATH, 'r') as f:
                            checkpoint_data = json.load(f)
                            best_params = checkpoint_data.get('best_params', {})
                            tuning_results = checkpoint_data.get('tuning_results', {})
                    except (IOError, json.JSONDecodeError) as e:
                        print(f"Error loading tuning checkpoint: {e}")
                        best_params = {}
                        tuning_results = {}
            else:
                # Create new model state
                model_state = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset_size": len(transformed_features),
                    "feature_count": transformed_features.shape[1],
                    "anomaly_percentage": float((sum(y_true)/len(y_true))*100),
                    "status": "starting_tuning"
                }
                
                # Save initial model state with proper error handling
                try:
                    with open(MODEL_STATE_PATH, 'w') as f:
                        json.dump(model_state, f, indent=4)
                    print(f"Initial model state saved to {MODEL_STATE_PATH}")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving initial model state: {e}")
                
                # Perform hyperparameter tuning with progress tracking and checkpointing
                print("\nStarting hyperparameter tuning with memory optimization...")
                best_params, tuning_results = model_tuner.tune_models(transformed_features, y_true, checkpoint_path=TUNING_CHECKPOINT_PATH)
                
                # Update model state after tuning with proper error handling
                model_state["status"] = "tuning_completed"
                model_state["tuning_completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                model_state["best_params"] = best_params
                
                # Serialize numpy values to make them JSON compatible
                serializable_best_params = {}
                for model_name, params in best_params.items():
                    serializable_best_params[model_name] = {}
                    for param_name, param_value in params.items():
                        if isinstance(param_value, (np.float32, np.float64, np.int32, np.int64)):
                            serializable_best_params[model_name][param_name] = float(param_value) if isinstance(param_value, (np.float32, np.float64)) else int(param_value)
                        else:
                            serializable_best_params[model_name][param_name] = param_value
                
                model_state["best_params"] = serializable_best_params
                
                try:
                    with open(MODEL_STATE_PATH, 'w') as f:
                        json.dump(model_state, f, indent=4)
                    print(f"Updated model state saved to {MODEL_STATE_PATH}")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving updated model state: {e}")
            
            # Clean up memory before learning curve analysis
            gc.collect()
            
            # Check if we need to run learning curve analysis
            if model_state.get('status') != 'learning_curves_completed':
                print("\nAnalyzing learning curves with memory optimization...")
                learning_curve_results = model_tuner.analyze_learning_curves(transformed_features, y_true, checkpoint_path=LEARNING_CURVES_CHECKPOINT_PATH)
                
                # Update model state after learning curve analysis with proper error handling
                model_state["status"] = "learning_curves_completed"
                model_state["learning_curves_completed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Serialize learning curve results to make them JSON compatible
                serializable_learning_curve_results = {}
                for model_name, results in learning_curve_results.items():
                    serializable_learning_curve_results[model_name] = {}
                    for key, value in results.items():
                        if isinstance(value, (list, np.ndarray)):
                            # Convert numpy arrays to lists and ensure all elements are native Python types
                            serializable_learning_curve_results[model_name][key] = [
                                float(x) if isinstance(x, (np.float32, np.float64)) else 
                                int(x) if isinstance(x, (np.int32, np.int64)) else x 
                                for x in value
                            ]
                        elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                            # Convert numpy scalars to native Python types
                            serializable_learning_curve_results[model_name][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                        else:
                            serializable_learning_curve_results[model_name][key] = value
                
                # Store serialized results in model state
                model_state["learning_curve_results_summary"] = serializable_learning_curve_results
                
                try:
                    with open(MODEL_STATE_PATH, 'w') as f:
                        json.dump(model_state, f, indent=4)
                    print(f"Updated model state with learning curve results saved to {MODEL_STATE_PATH}")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving updated model state with learning curve results: {e}")
            else:
                print("\nSkipping learning curve analysis - already completed according to model state")
                # Try to load learning curve results from checkpoint with proper error handling
                if os.path.exists(LEARNING_CURVES_CHECKPOINT_PATH):
                    try:
                        with open(LEARNING_CURVES_CHECKPOINT_PATH, 'r') as f:
                            checkpoint_data = json.load(f)
                            learning_curve_results = checkpoint_data.get('learning_curve_results', {})
                            print(f"Loaded learning curve results from checkpoint")
                            
                            # Print summary of loaded results
                            if learning_curve_results:
                                print("\nLoaded learning curve results summary:")
                                for model_name in learning_curve_results.keys():
                                    completion_time = learning_curve_results[model_name].get('completion_time', 'unknown')
                                    print(f"  - {model_name}: Completed at {completion_time}")
                            else:
                                print("No learning curve results found in checkpoint")
                    except (IOError, json.JSONDecodeError) as e:
                        print(f"Error loading learning curve checkpoint: {e}")
                        learning_curve_results = {}
                else:
                    print(f"No learning curve checkpoint found at {LEARNING_CURVES_CHECKPOINT_PATH}")
                    learning_curve_results = {}
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
    
    # Create or load detection state file to track progress
    detection_state_path = os.path.join(DRIVE_OUTPUT_PATH, "detection_state.json")
    
    # Check if detection state file exists and load it
    if os.path.exists(detection_state_path):
        try:
            with open(detection_state_path, 'r') as f:
                detection_state = json.load(f)
                print(f"\nLoaded existing detection state from {detection_state_path}")
                print(f"Status: {detection_state.get('status', 'unknown')}")
                print(f"Completed models: {len(detection_state.get('completed_models', []))}/{len(model_configs)}")
                print(f"Last updated: {detection_state.get('last_updated', detection_state.get('timestamp', 'unknown'))}")
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading detection state: {e}")
            # Create new detection state if loading fails
            detection_state = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "models": [config['model_type'] for config in model_configs],
                "completed_models": [],
                "status": "starting_detection",
                "progress": {
                    "total_models": len(model_configs),
                    "completed_models": 0
                }
            }
            # Save new detection state
            with open(detection_state_path, 'w') as f:
                json.dump(detection_state, f, indent=4)
            print(f"\nDetection state initialized and saved to {detection_state_path}")
    else:
        # Create new detection state if file doesn't exist
        detection_state = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models": [config['model_type'] for config in model_configs],
            "completed_models": [],
            "status": "starting_detection",
            "progress": {
                "total_models": len(model_configs),
                "completed_models": 0
            }
        }
        # Save new detection state
        with open(detection_state_path, 'w') as f:
            json.dump(detection_state, f, indent=4)
        print(f"\nDetection state initialized and saved to {detection_state_path}")
    
    try:
        with memory_cleanup():
            for config in model_configs:
                print(f"\nRunning anomaly detection with {config['model_type']}...")
                # Initialize detector_params with only the parameters that AnomalyDetector accepts directly
                detector_params = {'model_type': config['model_type']}
                
                # Add contamination if present in config
                if 'contamination' in config:
                    detector_params['contamination'] = config['contamination']
                
                # Add model-specific parameters based on model type
                if config['model_type'] == 'isolation_forest':
                    # Don't pass these to constructor, IsolationForest is created inside AnomalyDetector
                    pass
                elif config['model_type'] == 'lof' and 'params' in config:
                    if 'lof_neighbors' in config['params']:
                        detector_params['lof_neighbors'] = config['params']['lof_neighbors']
                elif config['model_type'] == 'one_class_svm' and 'params' in config:
                    if 'svm_nu' in config['params']:
                        detector_params['svm_nu'] = config['params']['svm_nu']
                    if 'svm_kernel' in config['params']:
                        detector_params['svm_kernel'] = config['params']['svm_kernel']
                    if 'svm_gamma' in config['params']:
                        detector_params['svm_gamma'] = config['params']['svm_gamma']
                elif config['model_type'] == 'hbos' and 'params' in config:
                    if 'hbos_n_bins' in config['params']:
                        detector_params['hbos_n_bins'] = config['params']['hbos_n_bins']
                    if 'hbos_alpha' in config['params']:
                        detector_params['hbos_alpha'] = config['params']['hbos_alpha']
                elif config['model_type'] == 'birch':
                    # Birch parameters are not directly passed to AnomalyDetector constructor
                    # They will be handled internally by the AnomalyDetector class
                    pass
                
                # Check if this model has already been processed by looking at detection state
                if config['model_type'] in detection_state.get('completed_models', []):
                    print(f"\nSkipping {config['model_type']} - already processed according to checkpoint")
                    
                    # Load the saved results for this model
                    model_results_path = os.path.join(DRIVE_OUTPUT_PATH, f"anomaly_results_{config['model_type']}.json")
                    if os.path.exists(model_results_path):
                        try:
                            with open(model_results_path, 'r') as f:
                                model_data = json.load(f)
                                all_anomalies_results[config['model_type']] = np.array(model_data['anomalies'])
                                model_metrics[config['model_type']] = model_data['metrics']
                                print(f"Loaded saved results for {config['model_type']} from {model_results_path}")
                                continue
                        except (IOError, json.JSONDecodeError) as e:
                            print(f"Error loading saved results, will reprocess: {e}")
                    else:
                        print(f"No saved results found, will reprocess {config['model_type']}")
                
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
                    "Anomaly_Percentage": float((sum(anomalies)/len(anomalies))*100),
                    "Completion_Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
                # Print results for this model
                print(f"Results for {config['model_type']}:")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1 Score: {f1:.4f}")
                print(f"  Anomalies detected: {sum(anomalies)}")
                print(f"  Anomaly percentage: {(sum(anomalies)/len(anomalies))*100:.2f}%")
                
                # Save individual model results to JSON file
                model_results_path = os.path.join(DRIVE_OUTPUT_PATH, f"anomaly_results_{config['model_type']}.json")
                model_data = {
                    "model_type": config['model_type'],
                    "anomalies": anomalies.tolist(),  # Convert numpy array to list for JSON serialization
                    "metrics": model_metrics[config['model_type']],
                    "params": config.get('params', {}),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                try:
                    with open(model_results_path, 'w') as f:
                        json.dump(model_data, f, indent=4)
                    print(f"Model results saved to {model_results_path}")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving results for {config['model_type']}: {e}")
                
                # Update detection state
                if config['model_type'] not in detection_state.get('completed_models', []):
                    detection_state["completed_models"].append(config['model_type'])
                detection_state["progress"]["completed_models"] = len(detection_state["completed_models"])
                detection_state["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                try:
                    with open(detection_state_path, 'w') as f:
                        json.dump(detection_state, f, indent=4)
                    print(f"Detection state updated: {len(detection_state['completed_models'])}/{len(model_configs)} models completed")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving detection state: {e}")
                
                # Clean up memory after each model evaluation
                gc.collect()
    except Exception as e:
        print(f"Error during anomaly detection: {str(e)}")
        raise
    finally:
        # Final memory cleanup
        gc.collect()

    # Update detection state to mark completion
    detection_state["status"] = "completed"
    detection_state["completion_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(detection_state_path, 'w') as f:
        json.dump(detection_state, f, indent=4)
    print(f"\nDetection process completed and state saved to {detection_state_path}")
    
    # Save final results to file
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