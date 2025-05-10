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
        # Remove zero-variance features
        features = self.variance_threshold.fit_transform(features)
        
        # Scale the features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit the anomaly detector
        if self.model_type != 'dbscan':
            self.detector.fit(scaled_features)
        else:
            self.scaled_features_for_dbscan_ = scaled_features
    
    def detect_anomalies(self, features):
        # Remove zero-variance features (using the same threshold as in fit)
        features = self.variance_threshold.transform(features)
        
        if self.model_type == 'dbscan':
            predictions = self.detector.fit_predict(self.scaled_features_for_dbscan_)
        else:
            scaled_features = self.scaler.transform(features)
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
            'dbscan_eps': self.dbscan_eps,
            'dbscan_min_samples': self.dbscan_min_samples,
            'hbos_n_bins': self.hbos_n_bins,
            'hbos_alpha': self.hbos_alpha,
            'detector': self.detector,
            'scaler': self.scaler,
            'variance_threshold': self.variance_threshold,
            'scaled_features_for_dbscan_': getattr(self, 'scaled_features_for_dbscan_', None)
        }
        joblib.dump(model_data, filepath)
        print(f"Anomaly detector model ({self.model_type}) saved to {filepath}")

    @classmethod
    def load_model(cls, filepath):
        """Loads an anomaly detector model and its scaler from a file."""
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
        if 'scaled_features_for_dbscan_' in model_data:
            model.scaled_features_for_dbscan_ = model_data['scaled_features_for_dbscan_']
        print(f"Anomaly detector model ({model.model_type}) loaded from {filepath}")
        return model