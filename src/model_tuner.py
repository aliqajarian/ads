import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, learning_curve, ParameterGrid
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.hbos import HBOS
from sklearn.cluster import Birch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json
import os
from datetime import datetime

class ModelTuner:
    def __init__(self, output_dir):
        """
        Initialize the ModelTuner with output directory.
        
        Args:
            output_dir (str): Directory to save tuning results and models
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, "model_tuning_results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize base models with supported anomaly detectors
        self.base_models = {
            'isolation_forest': IsolationForest(),
            'lof': LocalOutlierFactor(novelty=True),
            'one_class_svm': OneClassSVM(),
            'hbos': HBOS(),
            'dbscan': DBSCAN()
        }
        
        # Define optimized parameter grids for each model
        self.param_grids = {
            'isolation_forest': {
                'n_estimators': [100],  # Reduced to single optimal value
                'max_samples': ['auto'],  # Reduced to default value
                'contamination': [0.1]  # Reduced to single optimal value
            },
            'lof': {
                'n_neighbors': [20],  # Single optimal value
                'contamination': [0.1],  # Single optimal value
                'metric': ['euclidean']  # Most common metric
            },
            'one_class_svm': {
                'kernel': ['rbf'],  # Most effective kernel
                'nu': [0.1],  # Single optimal value
                'gamma': ['scale']  # Default value
            },
            'hbos': {
                'n_bins': [10],  # Default value
                'alpha': [0.1],  # Single optimal value
                'contamination': [0.1]  # Single optimal value
            },
            'dbscan': {
                'eps': [0.5],  # Default value for neighborhood size
                'min_samples': [5],  # Default value for minimum samples in neighborhood
                'metric': ['euclidean']  # Most common metric
            }
        }

        
    def check_tuning_status(self, checkpoint_path=None):
        """
        Check which models have been tuned based on saved checkpoints.
        
        Args:
            checkpoint_path (str, optional): Path to the checkpoint file. If None, will look for the latest checkpoint.
            
        Returns:
            dict: A dictionary containing tuning status for each model and overall progress
        """
        if checkpoint_path is None:
            # Look for the latest checkpoint in results directory
            checkpoint_files = [f for f in os.listdir(self.results_dir) if f.startswith("tuning_checkpoint_")]
            if not checkpoint_files:
                return {
                    'tuned_models': [],
                    'untuned_models': list(self.base_models.keys()),
                    'progress': {
                        'total_models': len(self.base_models),
                        'completed_models': 0,
                        'remaining_models': len(self.base_models)
                    },
                    'checkpoint_path': None
                }
            checkpoint_path = os.path.join(self.results_dir, sorted(checkpoint_files)[-1])
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                tuning_results = checkpoint_data.get('tuning_results', {})
                
                # Get lists of tuned and untuned models
                tuned_models = list(tuning_results.keys())
                untuned_models = [model for model in self.base_models.keys() if model not in tuned_models]
                
                # Calculate progress
                progress = {
                    'total_models': len(self.base_models),
                    'completed_models': len(tuned_models),
                    'remaining_models': len(untuned_models)
                }
                
                # Get performance metrics for tuned models
                model_metrics = {}
                for model_name, results in tuning_results.items():
                    model_metrics[model_name] = {
                        'f1_score': results['best_score'],
                        'completion_time': results['completion_time'],
                        'best_parameters': results['best_parameters']
                    }
                
                return {
                    'tuned_models': tuned_models,
                    'untuned_models': untuned_models,
                    'progress': progress,
                    'model_metrics': model_metrics,
                    'checkpoint_path': checkpoint_path,
                    'last_updated': checkpoint_data.get('timestamp')
                }
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading checkpoint file: {e}")
            return {
                'tuned_models': [],
                'untuned_models': list(self.base_models.keys()),
                'progress': {
                    'total_models': len(self.base_models),
                    'completed_models': 0,
                    'remaining_models': len(self.base_models)
                },
                'error': str(e),
                'checkpoint_path': None
            }
        


    def preprocess_features(self, X):
        """
        Preprocess features to handle zero-variance and near-zero-variance features.
        
        Args:
            X: Feature matrix
            
        Returns:
            Preprocessed feature matrix and preprocessing info
        """
        # Check for zero and near-zero variance features
        feature_variances = np.var(X, axis=0)
        non_zero_variance_mask = feature_variances > 0
        near_zero_variance_mask = (feature_variances > 0) & (feature_variances < 1e-10)
        
        preprocessing_info = {
            'original_shape': X.shape,
            'non_zero_variance_count': int(np.sum(non_zero_variance_mask)),
            'near_zero_variance_count': int(np.sum(near_zero_variance_mask)),
            'zero_variance_count': int(np.sum(feature_variances == 0))
        }
        
        # If all features have zero variance, add small noise
        if preprocessing_info['non_zero_variance_count'] == 0:
            print("Warning: All features have zero variance. Adding small noise...")
            epsilon = 1e-6
            X = X + np.random.normal(0, epsilon, size=X.shape)
            preprocessing_info['noise_added'] = True
            preprocessing_info['noise_epsilon'] = epsilon
        else:
            # Keep only features with non-zero variance
            X = X[:, non_zero_variance_mask]
            preprocessing_info['noise_added'] = False
            preprocessing_info['selected_features_mask'] = non_zero_variance_mask
        
        return X, preprocessing_info

    def tune_models(self, X, y, checkpoint_path=None):
        """
        Perform hyperparameter tuning for all models.
        
        Args:
            X: Feature matrix
            y: Target labels
            checkpoint_path: Optional path to load/save checkpoints
            
        Returns:
            dict: Best parameters and scores for each model
        """
        print("\nPerforming hyperparameter tuning for all models...")
        
        # Preprocess features
        X_processed, preprocessing_info = self.preprocess_features(X)
        print("\nFeature preprocessing summary:")
        for key, value in preprocessing_info.items():
            print(f"  {key}: {value}")
        
        # First check tuning status to get latest results
        status = self.check_tuning_status(checkpoint_path)
        tuning_results = {}
        best_params = {}
        
        # Set default checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.results_dir, f"tuning_checkpoint_{self.timestamp}.json")
            
        # Try to load the most recent results first
        if status['checkpoint_path'] and os.path.exists(status['checkpoint_path']):
            print(f"Loading latest results from {status['checkpoint_path']}")
            try:
                with open(status['checkpoint_path'], 'r') as f:
                    checkpoint_data = json.load(f)
                    tuning_results = checkpoint_data.get('tuning_results', {})
                    best_params = checkpoint_data.get('best_params', {})
                    print("\nPreviously tuned models:")
                    for model in tuning_results:
                        print(f"  - {model}: F1={tuning_results[model]['best_score']:.4f}")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load checkpoint file: {e}")
                tuning_results = {}
                best_params = {}
        
        # Get list of models to tune (exclude already tuned models)
        models_to_tune = [model_name for model_name in self.base_models.keys()
                         if model_name not in tuning_results]
        
        if not models_to_tune:
            print("All models have been tuned. Loading results from checkpoint.")
            return best_params, tuning_results
        
        print(f"\nRemaining models to tune: {len(models_to_tune)}")
        for model_name in models_to_tune:
            print(f"  - {model_name}")
        
        for model_name in models_to_tune:
            print(f"\nTuning hyperparameters for {model_name}...")
            
            # Handle LOF and DBSCAN models differently
            if model_name in ['lof', 'dbscan']:
                # For LOF and Birch, we need to fit and predict separately
                best_score = -float('inf')
                best_model = None
                best_params_set = None
                cv_results = {'mean_test_score': [], 'params': []}
                
                # Manual grid search for LOF/Birch
                for params in ParameterGrid(self.param_grids[model_name]):
                    if model_name == 'lof':
                        model = LocalOutlierFactor(novelty=True, **params)
                        model.fit(X)
                        y_pred = model.predict(X)
                    else:  # Birch
                        model = Birch(**params)
                        y_pred = model.fit_predict(X)
                    
                    y_pred_binary = np.where(y_pred == -1, 1, 0)
                    score = f1_score(y, y_pred_binary, zero_division=1)
                    
                    # Store results for each parameter combination
                    cv_results['mean_test_score'].append(score)
                    cv_results['params'].append(params)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_params_set = params
                
                best_params[model_name] = best_params_set
                y_pred = best_model.predict(X)
                
                # Store results for each parameter combination
                cv_results['mean_test_score'] = np.array(cv_results['mean_test_score'])
                cv_results['params'] = cv_results['params']
                
                # Create a mock grid_search object to maintain consistency
                class MockGridSearch:
                    def __init__(self, cv_results, best_score, best_params):
                        self.cv_results_ = cv_results
                        self.best_score_ = best_score
                        self.best_params_ = best_params
                        self.best_estimator_ = best_model
                
                grid_search = MockGridSearch(
                    cv_results=cv_results,
                    best_score=best_score,
                    best_params=best_params_set
                )
            else:
                # For other models, use GridSearchCV with custom scoring
                def custom_f1_scorer(estimator, X, y):
                    if hasattr(estimator, 'fit_predict'):
                        y_pred = estimator.fit_predict(X)
                    else:
                        estimator.fit(X)
                        y_pred = estimator.predict(X)
                    # Convert -1 to 1 for anomaly detection
                    y_pred_binary = np.where(y_pred == -1, 1, 0)
                    return f1_score(y, y_pred_binary, zero_division=1)

                # Configure GridSearchCV with optimized parallel processing settings
                n_jobs = min(os.cpu_count() or 1, 6)  # Use up to 4 cores or available cores
                grid_search = GridSearchCV(
                    estimator=self.base_models[model_name],
                    param_grid=self.param_grids[model_name],
                    scoring=custom_f1_scorer,
                    cv=3,  # Increased CV folds for better validation
                    verbose=1,  # Show progress
                    error_score=0.0,  # Return 0.0 instead of raising error
                    n_jobs=n_jobs,  # Utilize multiple cores for parallel processing
                    pre_dispatch='2*n_jobs'  # Optimize job pre-dispatching
                )
                
                # Fit the grid search
                grid_search.fit(X, y)
                
                # Get best parameters and score
                best_params[model_name] = grid_search.best_params_
                best_score = grid_search.best_score_
                
                # Calculate additional metrics
                best_model = grid_search.best_estimator_
                y_pred = best_model.fit_predict(X)
            
            y_pred = np.where(y_pred == -1, 1, 0)  # Convert to binary (1 for anomaly)
            
            metrics = {
                'precision': precision_score(y, y_pred, zero_division=1),
                'recall': recall_score(y, y_pred, zero_division=1),
                'f1': f1_score(y, y_pred, zero_division=1),
                'roc_auc': roc_auc_score(y, y_pred) if len(np.unique(y)) > 1 else None
            }
            
            # Store results
            tuning_results[model_name] = {
                'best_parameters': best_params[model_name],
                'best_score': float(best_score),
                'metrics': metrics,
                'all_results': {
                    'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                    'params': grid_search.cv_results_['params']
                },
                'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Print detailed results
            print(f"\nResults for {model_name}:")
            print(f"  Best parameters: {best_params[model_name]}")
            print(f"  Best F1 score: {best_score:.4f}")
            print(f"  Metrics:")
            print(f"    - Precision: {metrics['precision']:.4f}")
            print(f"    - Recall: {metrics['recall']:.4f}")
            print(f"    - F1: {metrics['f1']:.4f}")
            if metrics['roc_auc'] is not None:
                print(f"    - ROC AUC: {metrics['roc_auc']:.4f}")
            
            # Save checkpoint after each model with proper serialization and error handling
            # First serialize numpy values to make them JSON compatible
            serializable_tuning_results = {}
            for model_key, model_result in tuning_results.items():
                serializable_tuning_results[model_key] = {}
                for result_key, result_value in model_result.items():
                    if result_key == 'all_results':
                        # Handle nested dictionary with arrays
                        serializable_tuning_results[model_key]['all_results'] = {
                            'mean_test_score': [float(score) if isinstance(score, (np.float32, np.float64)) else score 
                                               for score in model_result['all_results']['mean_test_score']],
                            'params': model_result['all_results']['params']
                        }
                    elif isinstance(result_value, (np.float32, np.float64, np.int32, np.int64)):
                        # Convert numpy scalars to native Python types
                        serializable_tuning_results[model_key][result_key] = float(result_value) if isinstance(result_value, (np.float32, np.float64)) else int(result_value)
                    elif isinstance(result_value, dict) and 'precision' in result_value:
                        # Handle metrics dictionary
                        serializable_tuning_results[model_key][result_key] = {
                            k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                            for k, v in result_value.items()
                        }
                    else:
                        serializable_tuning_results[model_key][result_key] = result_value
            
            # Serialize best_params
            serializable_best_params = {}
            for model_name, params in best_params.items():
                serializable_best_params[model_name] = {}
                for param_name, param_value in params.items():
                    if isinstance(param_value, (np.float32, np.float64, np.int32, np.int64)):
                        serializable_best_params[model_name][param_name] = float(param_value) if isinstance(param_value, (np.float32, np.float64)) else int(param_value)
                    else:
                        serializable_best_params[model_name][param_name] = param_value
            
            checkpoint_data = {
                'tuning_results': serializable_tuning_results,
                'best_params': serializable_best_params,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'progress': {
                    'total_models': len(self.base_models),
                    'completed_models': len(tuning_results),
                    'remaining_models': len(self.base_models) - len(tuning_results)
                }
            }
            
            try:
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=4)
                print(f"\nProgress saved to {checkpoint_path}")
                print(f"Progress: {len(tuning_results)}/{len(self.base_models)} models completed")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error saving checkpoint: {e}")
                # Try saving to an alternative location
                alt_checkpoint_path = os.path.join(self.results_dir, f"tuning_checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                try:
                    with open(alt_checkpoint_path, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                    print(f"Saved backup checkpoint to {alt_checkpoint_path}")
                except Exception as e2:
                    print(f"Failed to save backup checkpoint: {e2}")
            
            # Save intermediate results file with proper serialization and error handling
            intermediate_results_path = os.path.join(
                self.results_dir, 
                f"tuning_results_{model_name}_{self.timestamp}.json"
            )
            
            # Serialize model results to make them JSON compatible
            serializable_model_results = {}
            for key, value in tuning_results[model_name].items():
                if key == 'all_results':
                    # Handle nested dictionary with arrays
                    serializable_model_results['all_results'] = {
                        'mean_test_score': [float(score) if isinstance(score, (np.float32, np.float64)) else score 
                                           for score in tuning_results[model_name]['all_results']['mean_test_score']],
                        'params': tuning_results[model_name]['all_results']['params']
                    }
                elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    # Convert numpy scalars to native Python types
                    serializable_model_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                elif isinstance(value, dict) and 'precision' in value:
                    # Handle metrics dictionary
                    serializable_model_results[key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in value.items()
                    }
                else:
                    serializable_model_results[key] = value
            
            try:
                with open(intermediate_results_path, 'w') as f:
                    json.dump(serializable_model_results, f, indent=4)
                print(f"Detailed results for {model_name} saved to {intermediate_results_path}")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error saving intermediate results for {model_name}: {e}")
                # Try saving to an alternative location
                alt_results_path = os.path.join(
                    self.results_dir, 
                    f"tuning_results_{model_name}_{self.timestamp}_backup.json"
                )
                try:
                    with open(alt_results_path, 'w') as f:
                        json.dump(serializable_model_results, f, indent=4)
                    print(f"Saved backup results to {alt_results_path}")
                except Exception as e2:
                    print(f"Failed to save backup results: {e2}")
        
        # Save final tuning results
        self._save_tuning_results(tuning_results)
        
        return best_params, tuning_results

    def analyze_learning_curves(self, X, y, checkpoint_path=None):
        """
        Analyze learning curves for all models.
        
        Args:
            X: Feature matrix
            y: Target labels
            checkpoint_path: Optional path to load/save checkpoints
            
        Returns:
            dict: Learning curve results for each model
        """
        print("\nAnalyzing learning curves for all models...")
        
        learning_curve_results = {}
        
        # Set default checkpoint path if not provided
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.results_dir, f"learning_curves_checkpoint_{self.timestamp}.json")
        
        # Load checkpoint if exists with proper error handling
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                    learning_curve_results = checkpoint_data.get('learning_curve_results', {})
                    print("Loaded progress:")
                    for model in learning_curve_results:
                        print(f"  - {model}: Completed")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading checkpoint: {e}")
                learning_curve_results = {}
        
        # Get list of models to analyze (exclude already analyzed models from checkpoint)
        models_to_analyze = [model_name for model_name in self.base_models.keys()
                            if model_name not in learning_curve_results]
        
        if not models_to_analyze:
            print("Learning curves for all models have been analyzed. Loading results from checkpoint.")
            return learning_curve_results
        
        print(f"\nRemaining models to analyze: {len(models_to_analyze)}")
        for model_name in models_to_analyze:
            print(f"  - {model_name}")
        
        # Configure parallel processing with optimized settings
        n_jobs = min(os.cpu_count() or 1, 4)  # Use up to 4 cores or available cores
        # Set up memory management for parallel processing
        from joblib import parallel_backend
        with parallel_backend('threading', n_jobs=n_jobs):
            for model_name in models_to_analyze:
                print(f"\nCalculating learning curve for {model_name}...")
                
                # Initialize model from base_models
                model = self.base_models[model_name]
                
                # Define custom scorer for anomaly detection
                def custom_f1_scorer(estimator, X, y):
                    if hasattr(estimator, 'fit_predict'):
                        y_pred = estimator.fit_predict(X)
                    else:
                        estimator.fit(X)
                        y_pred = estimator.predict(X)
                    # Convert -1 to 1 for anomaly detection
                    y_pred_binary = np.where(y_pred == -1, 1, 0)
                    return f1_score(y, y_pred_binary, average='weighted', zero_division=1)

                # Optimize learning curve calculation based on model type
                if model_name == 'one_class_svm':
                    try:
                        print(f"  - Starting one_class_svm learning curve calculation with {X.shape[0]} samples...")
                        # Use minimal settings for one_class_svm to speed up calculation
                        train_sizes, train_scores, test_scores = learning_curve(
                            model, X, y,
                            cv=2,  # Minimal cross-validation
                            scoring=custom_f1_scorer,
                            train_sizes=np.linspace(0.4, 1.0, 3),  # Only 3 points with larger starting size
                            n_jobs=1,  # Single process
                            verbose=1,  # Enable progress tracking
                            error_score='raise'  # Raise errors instead of returning NaN
                        )
                        print("  - Successfully completed one_class_svm learning curve calculation")
                    except Exception as e:
                        print(f"Error calculating learning curve for one_class_svm: {str(e)}")
                        # Initialize empty results for failed calculation
                        train_sizes = np.array([])
                        train_scores = np.array([])
                        test_scores = np.array([])
                        # Save error information
                        learning_curve_results[model_name] = {
                            'error': str(e),
                            'status': 'failed',
                            'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        continue  # Skip to next model
                else:
                    # Standard settings for other models
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X, y,
                        cv=2,  # Minimal cross-validation for notebooks
                        scoring=custom_f1_scorer,
                        train_sizes=np.linspace(0.3, 1.0, 4),  # Fewer points to reduce memory usage
                        n_jobs=1,  # Single process for notebook stability
                        verbose=0  # Reduced output for cleaner notebook display
                    )
                
                # Calculate statistics
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
            
                # Store results
                learning_curve_results[model_name] = {
                    'train_sizes': train_sizes.tolist(),
                    'train_mean': train_mean.tolist(),
                    'train_std': train_std.tolist(),
                    'test_mean': test_mean.tolist(),
                    'test_std': test_std.tolist(),
                    'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            
                # Print progress
                print(f"\nLearning curve analysis completed for {model_name}")
                print(f"  Final train score: {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
                print(f"  Final test score: {test_mean[-1]:.4f} ± {test_std[-1]:.4f}")
                
                # Save checkpoint after each model with proper serialization and error handling
                # First serialize numpy values to make them JSON compatible
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
                
                checkpoint_data = {
                    'learning_curve_results': serializable_learning_curve_results,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                    'progress': {
                        'total_models': len(self.base_models),
                        'completed_models': len(learning_curve_results),
                        'remaining_models': len(self.base_models) - len(learning_curve_results)
                    }
                }
                
                try:
                    with open(checkpoint_path, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                    print(f"\nProgress saved to {checkpoint_path}")
                    print(f"Progress: {len(learning_curve_results)}/{len(self.base_models)} models completed")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving checkpoint: {e}")
                    # Try saving to an alternative location
                    alt_checkpoint_path = os.path.join(self.results_dir, f"learning_curves_checkpoint_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    try:
                        with open(alt_checkpoint_path, 'w') as f:
                            json.dump(checkpoint_data, f, indent=4)
                        print(f"Saved backup checkpoint to {alt_checkpoint_path}")
                    except Exception as e2:
                        print(f"Failed to save backup checkpoint: {e2}")
                
                # Save intermediate learning curve results file with proper serialization and error handling
                intermediate_results_path = os.path.join(
                    self.results_dir, 
                    f"learning_curves_{model_name}_{self.timestamp}.json"
                )
                
                # Serialize model results to make them JSON compatible
                serializable_model_results = {}
                for key, value in learning_curve_results[model_name].items():
                    if isinstance(value, (list, np.ndarray)):
                        # Convert numpy arrays to lists and ensure all elements are native Python types
                        serializable_model_results[key] = [
                            float(x) if isinstance(x, (np.float32, np.float64)) else 
                            int(x) if isinstance(x, (np.int32, np.int64)) else x 
                            for x in value
                        ]
                    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                        # Convert numpy scalars to native Python types
                        serializable_model_results[key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                    else:
                        serializable_model_results[key] = value
                
                try:
                    with open(intermediate_results_path, 'w') as f:
                        json.dump(serializable_model_results, f, indent=4)
                    print(f"Detailed results for {model_name} saved to {intermediate_results_path}")
                except (IOError, json.JSONDecodeError) as e:
                    print(f"Error saving intermediate learning curve results for {model_name}: {e}")
                    # Try saving to an alternative location
                    alt_results_path = os.path.join(
                        self.results_dir, 
                        f"learning_curves_{model_name}_{self.timestamp}_backup.json"
                    )
                    try:
                        with open(alt_results_path, 'w') as f:
                            json.dump(serializable_model_results, f, indent=4)
                        print(f"Saved backup learning curve results to {alt_results_path}")
                    except Exception as e2:
                        print(f"Failed to save backup learning curve results: {e2}")
        
        # Save final learning curve results
        self._save_learning_curve_results(learning_curve_results)
        
        return learning_curve_results

    def _save_tuning_results(self, results):
        """Save tuning results to JSON and CSV files with proper serialization and error handling."""
        # Save detailed results to JSON
        json_path = os.path.join(self.results_dir, f"tuning_results_final_{self.timestamp}.json")
        
        # Serialize results to make them JSON compatible
        serializable_results = {}
        for model_key, model_result in results.items():
            serializable_results[model_key] = {}
            for result_key, result_value in model_result.items():
                if result_key == 'all_results':
                    # Handle nested dictionary with arrays
                    serializable_results[model_key]['all_results'] = {
                        'mean_test_score': [float(score) if isinstance(score, (np.float32, np.float64)) else score 
                                           for score in model_result['all_results']['mean_test_score']],
                        'params': model_result['all_results']['params']
                    }
                elif isinstance(result_value, (np.float32, np.float64, np.int32, np.int64)):
                    # Convert numpy scalars to native Python types
                    serializable_results[model_key][result_key] = float(result_value) if isinstance(result_value, (np.float32, np.float64)) else int(result_value)
                elif isinstance(result_value, dict) and 'precision' in result_value:
                    # Handle metrics dictionary
                    serializable_results[model_key][result_key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in result_value.items()
                    }
                else:
                    serializable_results[model_key][result_key] = result_value
        
        # Use context managers for file operations with better error handling
        try:
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Tuning results saved to {json_path}")
        except IOError as e:
            print(f"Error saving tuning results to JSON: {e}")
            # Try saving to an alternative location
            alt_json_path = os.path.join(self.results_dir, f"tuning_results_final_{self.timestamp}_backup.json")
            try:
                with open(alt_json_path, 'w') as f:
                    json.dump(serializable_results, f, indent=4)
                print(f"Saved backup tuning results to {alt_json_path}")
                json_path = alt_json_path  # Use the backup path for CSV reference
            except Exception as e2:
                print(f"Failed to save backup tuning results: {e2}")
                return
            
        # Create summary DataFrame
        summary_data = []
        for model_name, model_results in results.items():
            summary_data.append({
                'Model': model_name,
                'Best_F1_Score': model_results['best_score'],
                'Precision': model_results['metrics']['precision'],
                'Recall': model_results['metrics']['recall'],
                'F1_Score': model_results['metrics']['f1'],
                'ROC_AUC': model_results['metrics']['roc_auc'],
                'Best_Parameters': str(model_results['best_parameters']),
                'Completion_Time': model_results.get('completion_time', 'N/A')
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, f"tuning_summary_final_{self.timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nFinal tuning results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print("\nModel Performance Summary:")
        for row in summary_data:
            print(f"\n{row['Model']}:")
            print(f"  F1 Score: {row['F1_Score']:.4f}")
            print(f"  Precision: {row['Precision']:.4f}")
            print(f"  Recall: {row['Recall']:.4f}")
            if row['ROC_AUC'] is not None:
                print(f"  ROC AUC: {row['ROC_AUC']:.4f}")
            print(f"  Completed: {row['Completion_Time']}")

    def _save_learning_curve_results(self, results):
        """Save learning curve results to JSON and CSV files with proper serialization and error handling."""
        # Save detailed results to JSON
        json_path = os.path.join(self.results_dir, f"learning_curves_final_{self.timestamp}.json")
        
        # Serialize results to make them JSON compatible
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {}
            for key, value in model_results.items():
                if isinstance(value, (list, np.ndarray)):
                    # Convert numpy arrays to lists and ensure all elements are native Python types
                    serializable_results[model_name][key] = [
                        float(x) if isinstance(x, (np.float32, np.float64)) else 
                        int(x) if isinstance(x, (np.int32, np.int64)) else x 
                        for x in value
                    ]
                elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                    # Convert numpy scalars to native Python types
                    serializable_results[model_name][key] = float(value) if isinstance(value, (np.float32, np.float64)) else int(value)
                else:
                    serializable_results[model_name][key] = value
        
        # Use context managers for file operations with better error handling
        try:
            with open(json_path, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            print(f"Learning curve results saved to {json_path}")
        except IOError as e:
            print(f"Error saving learning curve results to JSON: {e}")
            # Try saving to an alternative location
            alt_json_path = os.path.join(self.results_dir, f"learning_curves_final_{self.timestamp}_backup.json")
            try:
                with open(alt_json_path, 'w') as f:
                    json.dump(serializable_results, f, indent=4)
                print(f"Saved backup learning curve results to {alt_json_path}")
                json_path = alt_json_path  # Use the backup path for CSV reference
            except Exception as e2:
                print(f"Failed to save backup learning curve results: {e2}")
                return
        
        # Create summary DataFrame
        summary_data = []
        for model_name, model_results in results.items():
            # Calculate final performance metrics
            final_train_score = model_results['train_mean'][-1]
            final_test_score = model_results['test_mean'][-1]
            performance_gap = final_train_score - final_test_score
            
            summary_data.append({
                'Model': model_name,
                'Final_Train_Score': final_train_score,
                'Final_Test_Score': final_test_score,
                'Performance_Gap': performance_gap,
                'Train_Score_Std': model_results['train_std'][-1],
                'Test_Score_Std': model_results['test_std'][-1],
                'Completion_Time': model_results.get('completion_time', 'N/A')
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, f"learning_curves_summary_final_{self.timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nFinal learning curve results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")
        print("\nLearning Curve Summary:")
        for row in summary_data:
            print(f"\n{row['Model']}:")
            print(f"  Train Score: {row['Final_Train_Score']:.4f} ± {row['Train_Score_Std']:.4f}")
            print(f"  Test Score: {row['Final_Test_Score']:.4f} ± {row['Test_Score_Std']:.4f}")
            print(f"  Performance Gap: {row['Performance_Gap']:.4f}")
            print(f"  Completed: {row['Completion_Time']}")