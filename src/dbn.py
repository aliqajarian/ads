import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
from joblib import Parallel, delayed
from tqdm import tqdm
import time

class DeepBeliefNetwork(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layers_sizes=None, layer_configs=None, random_state=None, verbose=0, checkpoint_path_prefix=None, 
                 early_stopping=True, patience=3, n_jobs=-1, use_tqdm=True, feature_subset_ratio=1.0):
        self.hidden_layers_sizes = hidden_layers_sizes if hidden_layers_sizes is not None else [256, 128]
        self.layer_configs = layer_configs
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_path_prefix = checkpoint_path_prefix
        self.early_stopping = early_stopping
        self.patience = patience
        self.n_jobs = n_jobs
        self.use_tqdm = use_tqdm
        self.feature_subset_ratio = feature_subset_ratio
        self.rbms_ = []
        self.layer_scores_ = [] # To store training/validation scores for each RBM
        self.selected_feature_indices_ = None # To store selected feature indices for consistent transformation

    def _train_rbm_with_early_stopping(self, rbm, X, X_val=None, max_iter=20, batch_size=100, patience=3):
        """Train an RBM with early stopping based on validation score."""
        best_val_score = -np.inf
        no_improvement_count = 0
        best_weights = None
        best_visible_bias = None
        best_hidden_bias = None
        
        # Initialize progress bar if requested
        iterator = tqdm(range(max_iter)) if self.use_tqdm else range(max_iter)
        
        # Initial fit with 1 iteration to initialize the model
        rbm.n_iter = 1
        rbm.fit(X)
        
        for epoch in iterator:
            # Train for one epoch
            start_time = time.time()
            rbm.n_iter = 1
            rbm.fit(X)
            
            # Calculate validation score
            if X_val is not None and X_val.shape[0] > 0:
                val_score = rbm.score_samples(X_val).mean()
                
                if self.use_tqdm:
                    iterator.set_description(f"Val score: {val_score:.4f}")
                
                # Check for improvement
                if val_score > best_val_score:
                    best_val_score = val_score
                    no_improvement_count = 0
                    # Save best model state
                    best_weights = rbm.components_.copy()
                    best_visible_bias = rbm.intercept_visible_.copy()
                    best_hidden_bias = rbm.intercept_hidden_.copy()
                else:
                    no_improvement_count += 1
                
                # Early stopping check
                if self.early_stopping and no_improvement_count >= patience:
                    if self.verbose > 0:
                        print(f"Early stopping at epoch {epoch+1} with best validation score: {best_val_score:.4f}")
                    break
            
            if self.verbose > 1:
                print(f"Epoch {epoch+1}/{max_iter} completed in {time.time() - start_time:.2f}s")
        
        # Restore best model if we have validation data and found a better model
        if X_val is not None and X_val.shape[0] > 0 and best_weights is not None:
            rbm.components_ = best_weights
            rbm.intercept_visible_ = best_visible_bias
            rbm.intercept_hidden_ = best_hidden_bias
        
        return rbm, best_val_score

    def _select_feature_subset(self, X, ratio=1.0, store_indices=False):
        """Select a subset of features to speed up training."""
        if ratio >= 1.0:
            self.selected_feature_indices_ = None
            return X
        
        n_features = X.shape[1]
        n_selected = max(1, int(n_features * ratio))
        
        if store_indices or self.selected_feature_indices_ is None:
            # Randomly select features (columns)
            np.random.seed(self.random_state if self.random_state is not None else 42)
            self.selected_feature_indices_ = np.random.choice(n_features, n_selected, replace=False)
        
        return X[:, self.selected_feature_indices_]

    def fit(self, X, y=None, X_val=None):
        start_time = time.time()
        self.rbms_ = []
        self.layer_scores_ = []
        
        # Apply feature subset selection if ratio < 1.0
        if self.feature_subset_ratio < 1.0:
            if self.verbose > 0:
                print(f"Using {self.feature_subset_ratio*100:.1f}% of features for faster training")
            current_input = self._select_feature_subset(X, self.feature_subset_ratio, store_indices=True)
            current_input_val = self._select_feature_subset(X_val, self.feature_subset_ratio) if X_val is not None else None
        else:
            current_input = X
            current_input_val = X_val
            self.selected_feature_indices_ = None

        if self.layer_configs is None:
            # Default configs with faster settings
            self.layer_configs = [{'n_iter': 10, 'learning_rate': 0.05, 'batch_size': 256} for _ in self.hidden_layers_sizes]
        elif len(self.layer_configs) != len(self.hidden_layers_sizes):
            raise ValueError("Length of layer_configs must match length of hidden_layers_sizes.")

        for i, n_hidden_components in enumerate(self.hidden_layers_sizes):
            layer_start_time = time.time()
            config = self.layer_configs[i]
            
            # Create RBM with optimized batch size
            rbm = BernoulliRBM(
                n_components=n_hidden_components,
                learning_rate=config.get('learning_rate', 0.05),  # Increased learning rate
                n_iter=1,  # We'll handle iterations manually for early stopping
                batch_size=config.get('batch_size', 256),  # Larger batch size
                random_state=self.random_state,
                verbose=0  # We'll handle verbosity ourselves
            )
            
            if self.verbose > 0:
                print(f"Training RBM layer {i+1}/{len(self.hidden_layers_sizes)} with {n_hidden_components} components...")

            # Train with early stopping
            rbm, val_score = self._train_rbm_with_early_stopping(
                rbm, 
                current_input, 
                current_input_val,
                max_iter=config.get('n_iter', 10),
                batch_size=config.get('batch_size', 256),
                patience=self.patience
            )
            
            self.rbms_.append(rbm)

            if self.checkpoint_path_prefix:
                try:
                    checkpoint_file = os.path.join(self.checkpoint_path_prefix, f"rbm_layer_{i+1}.pkl")
                    # Ensure the checkpoint directory exists
                    if os.path.dirname(checkpoint_file):
                        os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                    
                    joblib.dump(rbm, checkpoint_file)
                    if self.verbose > 0:
                        print(f"Saved checkpoint for RBM layer {i+1} to {checkpoint_file}")
                except Exception as e:
                    print(f"Warning: Failed to save checkpoint for RBM layer {i+1}: {str(e)}")
                    # Continue training even if checkpoint saving fails

            # Record scores (pseudo-likelihood)
            train_score = rbm.score_samples(current_input).mean()
            
            self.layer_scores_.append({'train_score': train_score, 'val_score': val_score})
            if self.verbose > 0:
                val_score_str = f"{val_score:.4f}" if val_score is not None else "N/A"
                print(f"RBM Layer {i+1}: Train Score: {train_score:.4f}, Val Score: {val_score_str}")
                print(f"Layer {i+1} training completed in {time.time() - layer_start_time:.2f}s")

            # Prepare input for the next layer - use parallel processing for large datasets
            if current_input.shape[0] > 10000 and self.n_jobs != 1:
                # Process in batches using parallel jobs
                batch_size = 1000
                n_batches = int(np.ceil(current_input.shape[0] / batch_size))
                
                def transform_batch(batch_idx):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, current_input.shape[0])
                    return rbm.transform(current_input[start_idx:end_idx])
                
                # Process batches in parallel
                results = Parallel(n_jobs=self.n_jobs)(delayed(transform_batch)(i) for i in range(n_batches))
                current_input = np.vstack(results)
                
                # Do the same for validation data if available
                if current_input_val is not None and current_input_val.shape[0] > 0:
                    n_val_batches = int(np.ceil(current_input_val.shape[0] / batch_size))
                    
                    def transform_val_batch(batch_idx):
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, current_input_val.shape[0])
                        return rbm.transform(current_input_val[start_idx:end_idx])
                    
                    val_results = Parallel(n_jobs=self.n_jobs)(delayed(transform_val_batch)(i) for i in range(n_val_batches))
                    current_input_val = np.vstack(val_results)
            else:
                # For smaller datasets, just transform directly
                current_input = rbm.transform(current_input)
                if current_input_val is not None and current_input_val.shape[0] > 0:
                    current_input_val = rbm.transform(current_input_val)
        
        if self.verbose > 0:
            print(f"Total DBN training completed in {time.time() - start_time:.2f}s")
        return self

    def transform(self, X):
        """Transform data through the DBN layers with parallel processing for large datasets."""
        # Apply feature selection if it was used during training
        if self.selected_feature_indices_ is not None:
            current_input = X[:, self.selected_feature_indices_]  # Use stored indices directly
        else:
            current_input = X
        
        for i, rbm in enumerate(self.rbms_):
            # Use parallel processing for large datasets
            if current_input.shape[0] > 10000 and self.n_jobs != 1:
                batch_size = 1000
                n_batches = int(np.ceil(current_input.shape[0] / batch_size))
                
                def transform_batch(batch_idx):
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, current_input.shape[0])
                    return rbm.transform(current_input[start_idx:end_idx])
                
                # Process batches in parallel
                results = Parallel(n_jobs=self.n_jobs)(delayed(transform_batch)(i) for i in range(n_batches))
                current_input = np.vstack(results)
            else:
                # For smaller datasets, just transform directly
                current_input = rbm.transform(current_input)
                
        return current_input

    def fit_transform(self, X, y=None, X_val=None):
        self.fit(X, y, X_val)
        return self.transform(X)

    def save_model(self, filepath):
        """Saves the trained DBN model to a file.
        
        Parameters
        ----------
        filepath : str
            Path where to save the model. For Google Colab, this should be a path in the mounted
            Google Drive directory (e.g., '/content/drive/MyDrive/models/dbn_model.pkl')
        """
        # Handle Google Drive paths and ensure the directory exists
        save_dir = os.path.dirname(filepath)
        if save_dir:
            # Create all parent directories if they don't exist
            os.makedirs(save_dir, exist_ok=True)
            
        model_data = {
            'hidden_layers_sizes': self.hidden_layers_sizes,
            'layer_configs': self.layer_configs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'rbms_': self.rbms_,
            'layer_scores_': self.layer_scores_,
            'checkpoint_path_prefix': self.checkpoint_path_prefix,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'n_jobs': self.n_jobs,
            'use_tqdm': self.use_tqdm,
            'feature_subset_ratio': self.feature_subset_ratio,
            'selected_feature_indices_': self.selected_feature_indices_
        }
        
        try:
            joblib.dump(model_data, filepath)
            if self.verbose > 0:
                print(f"DBN model saved to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {str(e)}")
            raise

    @classmethod
    def load_model(cls, filepath):
        """Loads a DBN model from a file.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model file. For Google Colab, this should be a path in the mounted
            Google Drive directory (e.g., '/content/drive/MyDrive/models/dbn_model.pkl')
        """
        try:
            model_data = joblib.load(filepath)
        except Exception as e:
            print(f"Error loading model from {filepath}: {str(e)}")
            raise
            
        # Compatibility for models saved before new parameters were added
        checkpoint_path_prefix = model_data.get('checkpoint_path_prefix', None)
        early_stopping = model_data.get('early_stopping', True)
        patience = model_data.get('patience', 3)
        n_jobs = model_data.get('n_jobs', -1)
        use_tqdm = model_data.get('use_tqdm', True)
        feature_subset_ratio = model_data.get('feature_subset_ratio', 1.0)
        selected_feature_indices = model_data.get('selected_feature_indices_', None)

        model = cls(
            hidden_layers_sizes=model_data['hidden_layers_sizes'],
            layer_configs=model_data['layer_configs'],
            random_state=model_data['random_state'],
            verbose=model_data['verbose'],
            checkpoint_path_prefix=checkpoint_path_prefix,
            early_stopping=early_stopping,
            patience=patience,
            n_jobs=n_jobs,
            use_tqdm=use_tqdm,
            feature_subset_ratio=feature_subset_ratio
        )
        model.rbms_ = model_data['rbms_']
        model.layer_scores_ = model_data['layer_scores_']
        model.selected_feature_indices_ = selected_feature_indices
        if model.verbose > 0:
            print(f"DBN model loaded from {filepath}")
        return model