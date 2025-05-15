import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from pyod.models.hbos import HBOS
import joblib
import os

class AnomalyDetector:
    def __init__(self, model_type='isolation_forest', contamination=0.1, lof_neighbors=20, 
                 svm_kernel='rbf', svm_gamma='scale', svm_nu=0.1, 
                 dbscan_eps=0.5, dbscan_min_samples=5,
                 hbos_n_bins=10, hbos_alpha=0.1):
        self.scaler = StandardScaler()
        self.variance_threshold = VarianceThreshold(threshold=0.0)
        self.model_type = model_type
        self.contamination = contamination
        self.lof_neighbors = lof_neighbors
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma
        self.svm_nu = svm_nu
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.hbos_n_bins = hbos_n_bins
        self.hbos_alpha = hbos_alpha
        self.detector = None
        
        # Initialize flags for feature processing status
        self.has_features = False  # Whether the model has been fitted with valid features
        self.has_variance = False  # Whether the features have sufficient variance
        self.original_n_features = None  # Original number of features before processing
        
        # Initialize zero variance handling attributes
        self.non_zero_variance_mask = None  # Mask for features with non-zero variance
        self.added_noise = False  # Whether noise was added to create variance

        if model_type == 'isolation_forest':
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        elif model_type == 'lof':
            self.detector = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=contamination, novelty=True)
        elif model_type == 'one_class_svm':
            self.detector = OneClassSVM(kernel=svm_kernel, gamma=svm_gamma, nu=svm_nu)
        elif model_type == 'dbscan':
            self.detector = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        elif model_type == 'hbos':
            self.detector = HBOS(
                n_bins=hbos_n_bins,
                alpha=hbos_alpha,
                contamination=contamination
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported types: 'isolation_forest', 'lof', 'one_class_svm', 'dbscan', 'hbos'")
    
    def fit(self, features):
        # Check if features array is empty
        if features.size == 0:
            print("Warning: Empty feature array provided to AnomalyDetector.fit()")
            self.has_features = False
            return
            
        try:
            # Store original feature shape for reference
            self.original_n_features = features.shape[1]
            
            # Check for zero variance features before applying threshold
            feature_variances = np.var(features, axis=0)
            non_zero_variance_mask = feature_variances > 0
            non_zero_variance_count = np.sum(non_zero_variance_mask)
            
            if non_zero_variance_count == 0:
                print("Warning: All features have zero variance. Using original features with a small epsilon added.")
                # Add small random noise to create variance when all features have zero variance
                epsilon = 1e-8
                features = features + np.random.normal(0, epsilon, size=features.shape)
                self.has_variance = False
                features_transformed = features
                # Store the fact that we added noise
                self.added_noise = True
            else:
                # Try to apply variance threshold normally
                try:
                    features_transformed = self.variance_threshold.fit_transform(features)
                    
                    # Check if any features remain after variance thresholding
                    if features_transformed.size == 0 or features_transformed.shape[1] == 0:
                        print("Warning: No features meet the variance threshold. Using features with non-zero variance.")
                        # Use only features with non-zero variance instead of all original features
                        features_transformed = features[:, non_zero_variance_mask]
                        self.non_zero_variance_mask = non_zero_variance_mask  # Store mask for prediction
                        self.has_variance = False
                    else:
                        self.has_variance = True
                        self.non_zero_variance_mask = None  # Not needed when threshold works
                except ValueError as ve:
                    print(f"Variance threshold error: {str(ve)}. Using features with non-zero variance.")
                    # Use only features with non-zero variance
                    features_transformed = features[:, non_zero_variance_mask]
                    self.non_zero_variance_mask = non_zero_variance_mask  # Store mask for prediction
                    self.has_variance = False
                    
                self.added_noise = False
                
            # Scale the features
            scaled_features = self.scaler.fit_transform(features_transformed)
            self.has_features = True
            
            # Fit the anomaly detector
            if self.model_type != 'dbscan':
                self.detector.fit(scaled_features)
            else:
                self.scaled_features_for_dbscan_ = scaled_features
                
        except Exception as e:
            print(f"Error during feature processing in fit(): {str(e)}")
            # Set flags to indicate the error state
            self.has_features = False
            self.has_variance = False
            # Re-raise with more context if needed
            raise ValueError(f"Failed to process features for anomaly detection: {str(e)}")
    
    def detect_anomalies(self, features):
        # Check if model was properly fitted with features
        if not hasattr(self, 'has_features') or not self.has_features:
            print("Warning: Model was not properly fitted with features. Returning all samples as normal.")
            return np.zeros(features.shape[0], dtype=bool)  # Return all as normal (non-anomalies)
            
        try:
            # Apply the same preprocessing steps as during fit
            if hasattr(self, 'added_noise') and self.added_noise:
                # Add the same small noise as during training if we had to do that
                epsilon = 1e-8
                features = features + np.random.normal(0, epsilon, size=features.shape)
                features_transformed = features
            elif hasattr(self, 'non_zero_variance_mask') and self.non_zero_variance_mask is not None:
                # Use the same non-zero variance feature mask as during training
                features_transformed = features[:, self.non_zero_variance_mask]
            elif hasattr(self, 'has_variance') and not self.has_variance:
                # For backward compatibility with older models
                features_transformed = features
            else:
                # Apply variance threshold transformation normally
                try:
                    features_transformed = self.variance_threshold.transform(features)
                except ValueError as ve:
                    print(f"Warning during prediction: {str(ve)}. Using original features.")
                    features_transformed = features
            
            # Scale the features
            scaled_features = self.scaler.transform(features_transformed)
            
            # Detect anomalies
            if self.model_type == 'dbscan':
                predictions = self.detector.fit_predict(self.scaled_features_for_dbscan_)
            else:
                predictions = self.detector.predict(scaled_features)
                
            return predictions == -1
            
        except Exception as e:
            print(f"Error during anomaly detection: {str(e)}")
            # Return a safe default (all normal) in case of error
            return np.zeros(features.shape[0], dtype=bool)  # Return all as normal (non-anomalies)

    def save_model(self, filepath):
        """Saves the trained anomaly detector model and its scaler to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        model_data = {
            'model_type': self.model_type,
            'contamination': self.contamination,
            'lof_neighbors': self.lof_neighbors,
            'svm_kernel': self.svm_kernel,
            'svm_gamma': self.svm_gamma,
            'svm_nu': self.svm_nu,
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples,
            'hbos_n_bins': self.hbos_n_bins,
            'hbos_alpha': self.hbos_alpha,
            'detector': self.detector,
            'scaler': self.scaler,
            'variance_threshold': self.variance_threshold,
            'scaled_features_for_dbscan_': getattr(self, 'scaled_features_for_dbscan_', None),
            # Save feature processing status flags
            'has_features': getattr(self, 'has_features', False),
            'has_variance': getattr(self, 'has_variance', False),
            'original_n_features': getattr(self, 'original_n_features', None),
            # Save new attributes for handling zero variance
            'non_zero_variance_mask': getattr(self, 'non_zero_variance_mask', None),
            'added_noise': getattr(self, 'added_noise', False)
        }
        try:
            joblib.dump(model_data, filepath)
            print(f"Anomaly detector model ({self.model_type}) saved to {filepath}")
        except Exception as e:
            print(f"Error saving anomaly detector model: {str(e)}")
            # Try to save with a different name as fallback
            alt_filepath = filepath + '.backup'
            try:
                joblib.dump(model_data, alt_filepath)
                print(f"Saved backup model to {alt_filepath}")
            except Exception as e2:
                print(f"Failed to save backup model: {str(e2)}")

    @classmethod
    def load_model(cls, filepath):
        """Loads an anomaly detector model and its scaler from a file."""
        try:
            model_data = joblib.load(filepath)
            model = cls(
                model_type=model_data['model_type'],
                contamination=model_data.get('contamination'),
                lof_neighbors=model_data.get('lof_neighbors', 20),
                svm_kernel=model_data.get('svm_kernel', 'rbf'),
                svm_gamma=model_data.get('svm_gamma', 'scale'),
                svm_nu=model_data.get('svm_nu', 0.1),
                dbscan_eps=model_data.get('dbscan_eps', 0.5),
                dbscan_min_samples=model_data.get('dbscan_min_samples', 5),
                hbos_n_bins=model_data.get('hbos_n_bins', 10),
                hbos_alpha=model_data.get('hbos_alpha', 0.1)
            )
            model.detector = model_data['detector']
            model.scaler = model_data['scaler']
            model.variance_threshold = model_data.get('variance_threshold', VarianceThreshold(threshold=0.0))
            
            # Load DBSCAN specific data if available
            if 'scaled_features_for_dbscan_' in model_data:
                model.scaled_features_for_dbscan_ = model_data['scaled_features_for_dbscan_']
            
            # Load feature processing status flags
            model.has_features = model_data.get('has_features', True)  # Default to True for backward compatibility
            model.has_variance = model_data.get('has_variance', True)  # Default to True for backward compatibility
            model.original_n_features = model_data.get('original_n_features', None)
            
            # Load zero variance handling attributes
            model.non_zero_variance_mask = model_data.get('non_zero_variance_mask', None)
            model.added_noise = model_data.get('added_noise', False)
            
            print(f"Anomaly detector model ({model.model_type}) loaded from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading anomaly detector model from {filepath}: {str(e)}")
            # Try to load from backup if exists
            backup_filepath = filepath + '.backup'
            if os.path.exists(backup_filepath):
                try:
                    print(f"Attempting to load from backup: {backup_filepath}")
                    return cls.load_model(backup_filepath)
                except Exception as e2:
                    print(f"Failed to load from backup: {str(e2)}")
            raise ValueError(f"Failed to load anomaly detector model: {str(e)}")