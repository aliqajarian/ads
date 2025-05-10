import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.hbos import HBOS
from sklearn.cluster import DBSCAN
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
        
        # Define base models
        self.base_models = {
            'isolation_forest': IsolationForest(random_state=42),
            'lof': LocalOutlierFactor(novelty=True),
            'one_class_svm': OneClassSVM(),
            'hbos': HBOS(),
            'dbscan': DBSCAN()
        }
        
        # Define parameter grids for each model with reduced search space
        self.param_grids = {
            'isolation_forest': {
                'n_estimators': [100],
                'contamination': [0.05],
                'max_samples': ['auto']
            },
            'lof': {
                'n_neighbors': [20],
                'contamination': [0.05],
                'metric': ['euclidean']
            },
            'one_class_svm': {
                'nu': [0.1],
                'kernel': ['rbf'],
                'gamma': ['scale']
            },
            'hbos': {
                'n_bins': [10],
                'alpha': [0.2],
                'contamination': [0.05]
            },
            'dbscan': {
                'eps': [0.5],
                'min_samples': [5],
                'metric': ['euclidean']
            }
        }

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
        
        tuning_results = {}
        best_params = {}
        
        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                tuning_results = checkpoint_data.get('tuning_results', {})
                best_params = checkpoint_data.get('best_params', {})
        
        # Get list of models to tune (exclude already tuned models from checkpoint)
        models_to_tune = [model_name for model_name in self.base_models.keys()
                         if model_name not in tuning_results]
        
        for model_name in models_to_tune:
            print(f"\nTuning hyperparameters for {model_name}...")
            
            # Create GridSearchCV object with reduced CV folds and early stopping
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grids[model_name],
                scoring='f1',
                cv=3,  # Reduced from 5 to 3 folds
                n_jobs=-1,
                verbose=1,
                error_score='raise'
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
                'precision': precision_score(y, y_pred),
                'recall': recall_score(y, y_pred),
                'f1': f1_score(y, y_pred),
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
                }
            }
            
            print(f"Best parameters for {model_name}: {best_params[model_name]}")
            print(f"Best F1 score: {best_score:.4f}")
            print(f"Additional metrics: {metrics}")
            
            # Save checkpoint after each model
            if checkpoint_path:
                checkpoint_data = {
                    'tuning_results': tuning_results,
                    'best_params': best_params,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=4)
                print(f"Checkpoint saved to {checkpoint_path}")
        
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
        
        # Load checkpoint if exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
                learning_curve_results = checkpoint_data.get('learning_curve_results', {})
        
        # Get list of models to analyze (exclude already analyzed models from checkpoint)
        models_to_analyze = [model_name for model_name in self.base_models.keys()
                            if model_name not in learning_curve_results]
        
        for model_name in models_to_analyze:
            print(f"\nCalculating learning curve for {model_name}...")
            
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y,
                cv=5,
                scoring='f1',
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1
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
                'test_std': test_std.tolist()
            }
            
            # Save checkpoint after each model
            if checkpoint_path:
                checkpoint_data = {
                    'learning_curve_results': learning_curve_results,
                    'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                }
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=4)
                print(f"Checkpoint saved to {checkpoint_path}")
        
        # Save final learning curve results
        self._save_learning_curve_results(learning_curve_results)
        
        return learning_curve_results

    def _save_tuning_results(self, results):
        """Save tuning results to JSON and CSV files."""
        # Save detailed results to JSON
        json_path = os.path.join(self.results_dir, f"tuning_results_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
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
                'Best_Parameters': str(model_results['best_parameters'])
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, f"tuning_summary_{self.timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nTuning results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")

    def _save_learning_curve_results(self, results):
        """Save learning curve results to JSON and CSV files."""
        # Save detailed results to JSON
        json_path = os.path.join(self.results_dir, f"learning_curves_{self.timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        
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
                'Test_Score_Std': model_results['test_std'][-1]
            })
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary_data)
        csv_path = os.path.join(self.results_dir, f"learning_curves_summary_{self.timestamp}.csv")
        summary_df.to_csv(csv_path, index=False)
        
        print(f"\nLearning curve results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  CSV: {csv_path}")