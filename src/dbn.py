import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os

class DeepBeliefNetwork(BaseEstimator, TransformerMixin):
    def __init__(self, hidden_layers_sizes=None, layer_configs=None, random_state=None, verbose=0, checkpoint_path_prefix=None):
        self.hidden_layers_sizes = hidden_layers_sizes if hidden_layers_sizes is not None else [256, 128]
        self.layer_configs = layer_configs
        self.random_state = random_state
        self.verbose = verbose
        self.checkpoint_path_prefix = checkpoint_path_prefix
        self.rbms_ = []
        self.layer_scores_ = [] # To store training/validation scores for each RBM

    def fit(self, X, y=None, X_val=None):
        self.rbms_ = []
        self.layer_scores_ = []
        current_input = X
        current_input_val = X_val

        if self.layer_configs is None:
            # Default configs if not provided
            self.layer_configs = [{'n_iter': 20, 'learning_rate': 0.01} for _ in self.hidden_layers_sizes]
        elif len(self.layer_configs) != len(self.hidden_layers_sizes):
            raise ValueError("Length of layer_configs must match length of hidden_layers_sizes.")

        for i, n_hidden_components in enumerate(self.hidden_layers_sizes):
            config = self.layer_configs[i]
            rbm = BernoulliRBM(
                n_components=n_hidden_components,
                learning_rate=config.get('learning_rate', 0.01),
                n_iter=config.get('n_iter', 20),
                batch_size=config.get('batch_size', 10),
                random_state=self.random_state,
                verbose=self.verbose
            )
            if self.verbose > 0:
                print(f"Training RBM layer {i+1}/{len(self.hidden_layers_sizes)} with {n_hidden_components} components...")

            rbm.fit(current_input)
            self.rbms_.append(rbm)

            if self.checkpoint_path_prefix:
                checkpoint_file = os.path.join(self.checkpoint_path_prefix, f"rbm_layer_{i+1}.pkl")
                os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
                joblib.dump(rbm, checkpoint_file)
                if self.verbose > 0:
                    print(f"Saved checkpoint for RBM layer {i+1} to {checkpoint_file}")

            # Record scores (pseudo-likelihood)
            train_score = rbm.score_samples(current_input).mean()
            val_score = None
            if current_input_val is not None and current_input_val.shape[0] > 0:
                val_score = rbm.score_samples(current_input_val).mean()
            
            self.layer_scores_.append({'train_score': train_score, 'val_score': val_score})
            if self.verbose > 0:
                print(f"RBM Layer {i+1}: Train Score (Pseudo-Likelihood): {train_score:.4f}")
                if val_score is not None:
                    print(f"RBM Layer {i+1}: Validation Score (Pseudo-Likelihood): {val_score:.4f}")

            # Prepare input for the next layer
            current_input = rbm.transform(current_input)
            if current_input_val is not None and current_input_val.shape[0] > 0:
                current_input_val = rbm.transform(current_input_val)
        
        return self

    def transform(self, X):
        current_input = X
        for rbm in self.rbms_:
            current_input = rbm.transform(current_input)
        return current_input

    def fit_transform(self, X, y=None, X_val=None):
        self.fit(X, y, X_val)
        return self.transform(X)

    def save_model(self, filepath):
        """Saves the trained DBN model to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'hidden_layers_sizes': self.hidden_layers_sizes,
            'layer_configs': self.layer_configs,
            'random_state': self.random_state,
            'verbose': self.verbose,
            'rbms_': self.rbms_,
            'layer_scores_': self.layer_scores_,
            'checkpoint_path_prefix': self.checkpoint_path_prefix # Save this too for consistency
        }
        joblib.dump(model_data, filepath)
        if self.verbose > 0:
            print(f"DBN model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Loads a DBN model from a file."""
        model_data = joblib.load(filepath)
        # Compatibility for models saved before checkpoint_path_prefix was added
        checkpoint_path_prefix = model_data.get('checkpoint_path_prefix', None) 

        model = cls(
            hidden_layers_sizes=model_data['hidden_layers_sizes'],
            layer_configs=model_data['layer_configs'],
            random_state=model_data['random_state'],
            verbose=model_data['verbose'],
            checkpoint_path_prefix=checkpoint_path_prefix
        )
        model.rbms_ = model_data['rbms_']
        model.layer_scores_ = model_data['layer_scores_']
        if model.verbose > 0:
            print(f"DBN model loaded from {filepath}")
        return model