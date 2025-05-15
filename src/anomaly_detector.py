import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN, Birch
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from pyod.models.hbos import HBOS
import joblib
import os

class AnomalyDetector:
    def __init__(self, model_type='isolation_forest', contamination=0.1, lof_neighbors=20, 
                 svm_kernel='rbf', svm_gamma='scale', svm_nu=0.1, 
                 dbscan_eps=0.5, dbscan_min_samples=5,
                 hbos_n_bins=10, hbos_alpha=0.1,
                 birch_threshold=0.5, birch_n_clusters=None, birch_branching_factor=50, birch_compute_labels=True,
                 debug_mode=False):  # Added debug mode parameter
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
        self.birch_threshold = birch_threshold
        self.birch_n_clusters = birch_n_clusters
        self.birch_branching_factor = birch_branching_factor
        self.birch_compute_labels = birch_compute_labels
        self.detector = None
        self.debug_mode = debug_mode  # Store debug mode setting
        
        # Initialize flags for feature processing status
        self.has_features = False  # Whether the model has been fitted with valid features
        self.has_variance = False  # Whether the features have sufficient variance
        self.original_n_features = None  # Original number of features before processing
        
        # Initialize zero variance handling attributes
        self.non_zero_variance_mask = None  # Mask for features with non-zero variance
        self.added_noise = False  # Whether noise was added to create variance
        self.noise_epsilon = 1e-6  # Default epsilon value for noise addition

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
        elif model_type == 'birch':
            self.detector = Birch(
                threshold=birch_threshold,
                n_clusters=birch_n_clusters,
                branching_factor=birch_branching_factor,
                compute_labels=birch_compute_labels
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported types: 'isolation_forest', 'lof', 'one_class_svm', 'dbscan', 'hbos', 'birch'")
    
    def inspect_features(self, features):
        """Inspect and validate features before fitting the model.
        
        This method analyzes the feature array to identify potential issues like:
        - Zero variance features
        - Features with very low variance
        - Constant or near-constant features
        - NaN or infinite values
        
        Args:
            features: numpy array of features to inspect
            
        Returns:
            dict: Dictionary with inspection results and statistics
        """
        if features.size == 0:
            return {
                'valid': False,
                'error': 'Empty feature array',
                'shape': (0, 0)
            }
            
        # Basic shape and content checks
        result = {
            'valid': True,
            'shape': features.shape,
            'n_samples': features.shape[0],
            'n_features': features.shape[1],
            'has_nan': np.isnan(features).any(),
            'has_inf': np.isinf(features).any(),
        }
        
        # Check for NaN or infinite values
        if result['has_nan'] or result['has_inf']:
            result['valid'] = False
            result['error'] = 'Features contain NaN or infinite values'
            nan_count = np.isnan(features).sum()
            inf_count = np.isinf(features).sum()
            result['nan_count'] = int(nan_count)
            result['inf_count'] = int(inf_count)
            return result
        
        # Compute feature statistics
        feature_variances = np.var(features, axis=0)
        feature_means = np.mean(features, axis=0)
        feature_mins = np.min(features, axis=0)
        feature_maxs = np.max(features, axis=0)
        feature_ranges = feature_maxs - feature_mins
        
        # Count features with zero and near-zero variance
        zero_var_mask = feature_variances == 0
        near_zero_var_mask = (feature_variances > 0) & (feature_variances < 1e-10)
        
        result.update({
            'mean_variance': float(np.mean(feature_variances)),
            'min_variance': float(np.min(feature_variances)),
            'max_variance': float(np.max(feature_variances)),
            'mean_range': float(np.mean(feature_ranges)),
            'zero_variance_count': int(np.sum(zero_var_mask)),
            'near_zero_variance_count': int(np.sum(near_zero_var_mask)),
            'non_zero_variance_count': int(np.sum(feature_variances > 0)),
            'zero_variance_percent': float(np.sum(zero_var_mask) / features.shape[1] * 100)
        })
        
        # Flag potential issues
        if result['zero_variance_count'] == features.shape[1]:
            result['issue'] = 'ALL_ZERO_VARIANCE'
            result['recommendation'] = 'All features have zero variance. Check feature extraction process.'
        elif result['zero_variance_count'] > features.shape[1] * 0.5:
            result['issue'] = 'MAJORITY_ZERO_VARIANCE'
            result['recommendation'] = 'Most features have zero variance. Consider different feature extraction.'
        elif result['zero_variance_count'] > 0:
            result['issue'] = 'SOME_ZERO_VARIANCE'
            result['recommendation'] = 'Some features have zero variance. Will be handled automatically.'
        
        # If in debug mode and there are constant features, show sample values
        if self.debug_mode and result['zero_variance_count'] > 0:
            constant_indices = np.where(zero_var_mask)[0][:5]  # Get up to 5 constant features
            constant_samples = {}
            for idx in constant_indices:
                constant_samples[f'feature_{idx}'] = float(features[0, idx])
            result['constant_feature_samples'] = constant_samples
        
        return result
        
    def fit(self, features):
        # Check if features array is empty
        if features.size == 0:
            print("Warning: Empty feature array provided to AnomalyDetector.fit()")
            self.has_features = False
            return
            
        # Inspect features if debug mode is enabled
        if self.debug_mode:
            inspection = self.inspect_features(features)
            print("Feature inspection results:")
            for key, value in inspection.items():
                if key != 'constant_feature_samples':
                    print(f"  {key}: {value}")
            if 'constant_feature_samples' in inspection:
                print("  Sample values from constant features:")
                for feat, val in inspection['constant_feature_samples'].items():
                    print(f"    {feat}: {val}")
            if 'issue' in inspection:
                print(f"  Issue detected: {inspection['issue']}")
                print(f"  Recommendation: {inspection['recommendation']}")

            
        try:
            # Store original feature shape for reference
            self.original_n_features = features.shape[1]
            
            # Enhanced feature statistics logging
            print(f"Feature array shape: {features.shape}")
            
            # Check for zero variance features before applying threshold
            feature_variances = np.var(features, axis=0)
            feature_means = np.mean(features, axis=0)
            feature_mins = np.min(features, axis=0)
            feature_maxs = np.max(features, axis=0)
            
            # Print feature statistics summary
            print(f"Feature statistics summary:")
            print(f"  Mean variance: {np.mean(feature_variances):.6f}")
            print(f"  Min variance: {np.min(feature_variances):.6f}")
            print(f"  Max variance: {np.max(feature_variances):.6f}")
            print(f"  Mean range (max-min): {np.mean(feature_maxs - feature_mins):.6f}")
            
            non_zero_variance_mask = feature_variances > 0
            non_zero_variance_count = np.sum(non_zero_variance_mask)
            print(f"  Features with non-zero variance: {non_zero_variance_count}/{features.shape[1]} ({non_zero_variance_count/features.shape[1]*100:.2f}%)")
            
            # If very few features have variance, print more details
            if non_zero_variance_count < features.shape[1] * 0.1 and features.shape[1] > 10:
                print("  Warning: Most features have zero variance. This may indicate an issue with feature extraction.")
                # Print sample of feature values for the first few constant features
                constant_features_indices = np.where(~non_zero_variance_mask)[0][:5]  # Get up to 5 constant features
                for idx in constant_features_indices:
                    print(f"  Constant feature #{idx}: all values = {features[0, idx]:.6f}")
            
            if non_zero_variance_count == 0:
                print("Warning: All features have zero variance. Using original features with noise added.")
                # Add small random noise to create variance when all features have zero variance
                # Using a slightly larger epsilon to ensure sufficient variance
                epsilon = 1e-6
                print(f"  Adding random noise with epsilon={epsilon} to create variance")
                
                # Save original features for reference
                features_original = features.copy()
                
                # Add noise and check if it created variance
                features = features + np.random.normal(0, epsilon, size=features.shape)
                
                # Verify the noise addition created variance
                new_variances = np.var(features, axis=0)
                new_non_zero_count = np.sum(new_variances > 0)
                print(f"  After noise addition: {new_non_zero_count}/{features.shape[1]} features have non-zero variance")
                
                # If noise didn't help, try a larger epsilon as a fallback
                if new_non_zero_count == 0:
                    epsilon = 1e-4
                    print(f"  First noise attempt failed. Trying larger epsilon={epsilon}")
                    features = features_original + np.random.normal(0, epsilon, size=features.shape)
                    
                    # Verify again
                    new_variances = np.var(features, axis=0)
                    new_non_zero_count = np.sum(new_variances > 0)
                    print(f"  After second noise attempt: {new_non_zero_count}/{features.shape[1]} features have non-zero variance")
                
                self.has_variance = False
                features_transformed = features
                # Store the fact that we added noise
                self.added_noise = True
                self.noise_epsilon = epsilon  # Store epsilon value for prediction
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
            if self.model_type == 'dbscan':
                self.scaled_features_for_dbscan_ = scaled_features
            elif self.model_type == 'birch':
                self.detector.fit(scaled_features)
                self.birch_labels_ = self.detector.predict(scaled_features)
                # Store the anomaly label (assuming the smallest cluster is anomalous)
                if len(np.unique(self.birch_labels_)) > 1:
                    # Find the smallest cluster and consider it as anomalies
                    cluster_sizes = np.bincount(self.birch_labels_)
                    self.anomaly_cluster_ = np.argmin(cluster_sizes)
                else:
                    # If only one cluster, no anomalies
                    self.anomaly_cluster_ = -1
            else:
                self.detector.fit(scaled_features)
                
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
                # Add the same noise as during training if we had to do that
                # Use the stored epsilon value if available, otherwise use default
                epsilon = getattr(self, 'noise_epsilon', 1e-6)
                print(f"Adding noise with epsilon={epsilon} during prediction (same as training)")
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
            elif self.model_type == 'birch':
                cluster_labels = self.detector.predict(scaled_features)
                # Mark samples in the anomaly cluster as anomalies
                if hasattr(self, 'anomaly_cluster_') and self.anomaly_cluster_ != -1:
                    predictions = (cluster_labels == self.anomaly_cluster_).astype(int) * -1
                else:
                    # If no anomaly cluster was identified, return all as normal
                    predictions = np.zeros(scaled_features.shape[0])
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
            'birch_threshold': self.birch_threshold,
            'birch_n_clusters': self.birch_n_clusters,
            'birch_branching_factor': self.birch_branching_factor,
            'birch_compute_labels': self.birch_compute_labels,
            'debug_mode': getattr(self, 'debug_mode', False),
            'detector': self.detector,
            'scaler': self.scaler,
            'variance_threshold': self.variance_threshold,
            'scaled_features_for_dbscan_': getattr(self, 'scaled_features_for_dbscan_', None),
            'birch_labels_': getattr(self, 'birch_labels_', None),
            'anomaly_cluster_': getattr(self, 'anomaly_cluster_', None),
            # Save feature processing status flags
            'has_features': getattr(self, 'has_features', False),
            'has_variance': getattr(self, 'has_variance', False),
            'original_n_features': getattr(self, 'original_n_features', None),
            # Save new attributes for handling zero variance
            'non_zero_variance_mask': getattr(self, 'non_zero_variance_mask', None),
            'added_noise': getattr(self, 'added_noise', False),
            'noise_epsilon': getattr(self, 'noise_epsilon', 1e-6)  # Save the epsilon value used for noise
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
                hbos_alpha=model_data.get('hbos_alpha', 0.1),
                birch_threshold=model_data.get('birch_threshold', 0.5),
                birch_n_clusters=model_data.get('birch_n_clusters', None),
                birch_branching_factor=model_data.get('birch_branching_factor', 50),
                birch_compute_labels=model_data.get('birch_compute_labels', True),
                debug_mode=model_data.get('debug_mode', False)
            )
            model.detector = model_data['detector']
            model.scaler = model_data['scaler']
            model.variance_threshold = model_data.get('variance_threshold', VarianceThreshold(threshold=0.0))
            
            # Load DBSCAN specific data if available
            if 'scaled_features_for_dbscan_' in model_data:
                model.scaled_features_for_dbscan_ = model_data['scaled_features_for_dbscan_']
                
            # Load BIRCH specific data if available
            if 'birch_labels_' in model_data:
                model.birch_labels_ = model_data['birch_labels_']
            if 'anomaly_cluster_' in model_data:
                model.anomaly_cluster_ = model_data['anomaly_cluster_']
            
            # Load feature processing status flags
            model.has_features = model_data.get('has_features', True)  # Default to True for backward compatibility
            model.has_variance = model_data.get('has_variance', True)  # Default to True for backward compatibility
            model.original_n_features = model_data.get('original_n_features', None)
            
            # Load zero variance handling attributes
            model.non_zero_variance_mask = model_data.get('non_zero_variance_mask', None)
            model.added_noise = model_data.get('added_noise', False)
            model.noise_epsilon = model_data.get('noise_epsilon', 1e-6)  # Load the epsilon value used for noise
            
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