import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib
import os

class AnomalyDetector:
    def __init__(self, model_type='isolation_forest', contamination=0.1, lof_neighbors=20, 
                 svm_kernel='rbf', svm_gamma='scale', svm_nu=0.1, 
                 ee_contamination=0.1, dbscan_eps=0.5, dbscan_min_samples=5):
        self.scaler = StandardScaler()
        self.model_type = model_type
        self.contamination = contamination
        self.lof_neighbors = lof_neighbors
        self.svm_kernel = svm_kernel
        self.svm_gamma = svm_gamma
        self.svm_nu = svm_nu
        self.ee_contamination = ee_contamination
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.detector = None # Initialize detector here

        if model_type == 'isolation_forest':
            self.detector = IsolationForest(contamination=contamination, random_state=42)
        elif model_type == 'lof':
            # For LOF, contamination is used to determine the threshold on novelty scores.
            # n_neighbors is a key parameter for LOF.
            self.detector = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=contamination, novelty=True)
        elif model_type == 'one_class_svm':
            # For OneClassSVM, nu is an upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
            # It's similar to contamination.
            self.detector = OneClassSVM(kernel=svm_kernel, gamma=svm_gamma, nu=svm_nu) # nu is often set similar to contamination
        elif model_type == 'elliptic_envelope':
            self.detector = EllipticEnvelope(contamination=ee_contamination, random_state=42)
        elif model_type == 'dbscan':
            # DBSCAN identifies outliers as points not belonging to any cluster (labeled as -1).
            # It doesn't use 'contamination' directly in its constructor but its parameters (eps, min_samples) influence outlier detection.
            self.detector = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Supported types: 'isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope', 'dbscan'")
    
    def fit(self, features):
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        # Fit the anomaly detector
        # DBSCAN is different as it doesn't have a separate predict method for new data in the same way for anomaly detection;
        # it assigns labels during fit. We'll handle its prediction in detect_anomalies.
        if self.model_type != 'dbscan':
            self.detector.fit(scaled_features)
        else:
            # For DBSCAN, fit_predict is typically used, and we'll store the scaled features for detect_anomalies
            self.scaled_features_for_dbscan_ = scaled_features
    
    def detect_anomalies(self, features):
        # Ensure features are scaled consistently with how the model was fit
        # For DBSCAN, we use the features it was 'fit' on (or rather, fit_predict would be called on)
        if self.model_type == 'dbscan':
            # DBSCAN's fit_predict assigns labels. -1 indicates an outlier.
            # We call fit_predict here as DBSCAN doesn't separate fit and predict for outlier detection in the same way as others.
            # This means DBSCAN is refit every time detect_anomalies is called if used this way, which is not ideal for performance
            # but necessary if new data is passed to detect_anomalies each time.
            # A more common use of DBSCAN for anomaly detection on *new* data would require a more complex setup
            # (e.g., fitting on a reference set and then checking new points' distances to clusters).
            # For simplicity in this context, we assume detect_anomalies is called with the same data used for 'fitting' (or the full dataset).
            # If features passed to detect_anomalies can be different from fit, this needs adjustment.
            # Using the stored scaled features from fit method for DBSCAN:
            predictions = self.detector.fit_predict(self.scaled_features_for_dbscan_)
        else:
            scaled_features = self.scaler.transform(features)
            # Predict anomalies (-1 for anomalies, 1 for normal for IsolationForest, OneClassSVM, LOF, EllipticEnvelope)
            predictions = self.detector.predict(scaled_features)
        return predictions == -1

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
            'ee_contamination': self.ee_contamination,
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples,
            'detector': self.detector,
            'scaler': self.scaler,
            # For DBSCAN, store relevant attributes if needed for predict/detect_anomalies on new data
            'scaled_features_for_dbscan_': getattr(self, 'scaled_features_for_dbscan_', None) 
        }
        joblib.dump(model_data, filepath)
        print(f"Anomaly detector model ({self.model_type}) saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Loads an anomaly detector model and its scaler from a file."""
        model_data = joblib.load(filepath)
        # Re-initialize the class with stored parameters
        model = cls(
            model_type=model_data['model_type'],
            contamination=model_data.get('contamination'), # Use .get for backward compatibility if some params were not saved
            lof_neighbors=model_data.get('lof_neighbors', 20),
            svm_kernel=model_data.get('svm_kernel', 'rbf'),
            svm_gamma=model_data.get('svm_gamma', 'scale'),
            svm_nu=model_data.get('svm_nu', 0.1),
            ee_contamination=model_data.get('ee_contamination', 0.1),
            dbscan_eps=model_data.get('dbscan_eps', 0.5),
            dbscan_min_samples=model_data.get('dbscan_min_samples', 5)
        )
        model.detector = model_data['detector']
        model.scaler = model_data['scaler']
        if 'scaled_features_for_dbscan_' in model_data:
            model.scaled_features_for_dbscan_ = model_data['scaled_features_for_dbscan_']
        print(f"Anomaly detector model ({model.model_type}) loaded from {filepath}")
        return model