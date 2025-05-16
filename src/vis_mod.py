#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enhanced Visualization Module Demo
This script demonstrates how to use the Visualizer class to generate comprehensive visualizations
for anomaly detection analysis, with clear comparisons across different models.
"""

import pandas as pd
import numpy as np
from src.visualizer import Visualizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.hbos import HBOS
from sklearn.cluster import DBSCAN
import os
from datetime import datetime, timedelta
import json

def create_sample_data(n_samples=1000):
    """Create sample data for visualization demonstration."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate sample review scores (1-5)
    scores = np.random.randint(1, 6, n_samples)
    
    # Generate sample review text
    review_texts = [f"This is a sample review text {i}" * np.random.randint(1, 5) for i in range(n_samples)]
    
    # Generate sample helpfulness ratios (0-1)
    helpfulness_ratios = np.random.uniform(0, 1, n_samples)
    
    # Generate sample user IDs
    user_ids = [f"user_{np.random.randint(1, 100)}" for _ in range(n_samples)]
    
    # Generate sample review times
    base_time = datetime.now()
    review_times = [base_time - timedelta(days=np.random.randint(0, 365)) for _ in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'review/score': scores,
        'review/text': review_texts,
        'helpfulness_ratio': helpfulness_ratios,
        'User_id': user_ids,
        'review_time': review_times
    })
    
    # Generate sample features
    features = np.random.randn(n_samples, 50)
    
    # Generate sample anomalies (10% of data)
    anomalies = np.zeros(n_samples, dtype=bool)
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    anomalies[anomaly_indices] = True
    
    # Generate sample model results
    model_results = {
        'isolation_forest': anomalies.copy(),
        'lof': np.random.choice([True, False], size=n_samples, p=[0.1, 0.9]),
        'one_class_svm': np.random.choice([True, False], size=n_samples, p=[0.12, 0.88]),
        'hbos': np.random.choice([True, False], size=n_samples, p=[0.08, 0.92])
    }
    
    # Generate sample DBN layer scores
    layer_scores = [
        {'train_score': 0.85, 'val_score': 0.82},
        {'train_score': 0.88, 'val_score': 0.84},
        {'train_score': 0.90, 'val_score': 0.85}
    ]
    
    return df, features, anomalies, model_results, layer_scores

def create_output_directories():
    """Create necessary output directories for visualizations."""
    base_dir = "ads_output"
    dirs = {
        'base': base_dir,
        'tsne': os.path.join(base_dir, "tsne_visualizations"),
        'correlation': os.path.join(base_dir, "correlation_visualizations"),
        'learning_curves': os.path.join(base_dir, "learning_curves"),
        'plots': os.path.join(base_dir, "plots")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created/verified directory: {dir_path}")
    
    return dirs

def load_model_metrics(results_dir):
    """Load model metrics from saved JSON files.
    
    Args:
        results_dir (str): Directory containing the model results JSON files
        
    Returns:
        dict: Dictionary of model metrics
    """
    model_metrics = {}
    try:
        # Look for files matching the pattern "anomaly_results_*.json"
        result_files = [f for f in os.listdir(results_dir) if f.startswith("anomaly_results_") and f.endswith(".json")]
        
        for file in result_files:
            model_type = file.replace("anomaly_results_", "").replace(".json", "")
            file_path = os.path.join(results_dir, file)
            
            try:
                with open(file_path, 'r') as f:
                    model_data = json.load(f)
                    metrics = model_data.get('metrics', {})
                    
                    # Convert model type to display name
                    display_name = model_type.replace('_', ' ').title()
                    
                    model_metrics[display_name] = {
                        'Precision': metrics.get('Precision', 0.0),
                        'Recall': metrics.get('Recall', 0.0),
                        'F1': metrics.get('F1', 0.0),
                        'Anomaly_Percentage': metrics.get('Anomaly_Percentage', 0.0)
                    }
                print(f"Loaded metrics for {display_name} from {file}")
            except (IOError, json.JSONDecodeError) as e:
                print(f"Error loading metrics for {model_type}: {e}")
        
        if not model_metrics:
            print("Warning: No model metrics found in JSON files. Using sample metrics.")
            # Use sample metrics as fallback
            model_metrics = {
                'Isolation Forest': {
                    'Precision': 0.85,
                    'Recall': 0.82,
                    'F1': 0.83,
                    'Anomaly_Percentage': 10.5
                },
                'Local Outlier Factor': {
                    'Precision': 0.78,
                    'Recall': 0.75,
                    'F1': 0.76,
                    'Anomaly_Percentage': 9.8
                },
                'One-Class SVM': {
                    'Precision': 0.82,
                    'Recall': 0.79,
                    'F1': 0.80,
                    'Anomaly_Percentage': 11.2
                },
                'HBOS': {
                    'Precision': 0.80,
                    'Recall': 0.77,
                    'F1': 0.78,
                    'Anomaly_Percentage': 10.8
                }
            }
        
        return model_metrics
    except Exception as e:
        print(f"Error loading model metrics: {e}")
        return {}

def generate_visualizations(df, features, anomalies, model_results, layer_scores, output_dirs):
    """Generate and save all visualizations."""
    visualizer = Visualizer()
    
    try:
        # Load model metrics from JSON files
        print("\nLoading model metrics from JSON files...")
        model_metrics = load_model_metrics(output_dirs['base'])
        
        # Plot t-SNE visualization for each model with enhanced labels
        print("\nGenerating t-SNE visualizations...")
        for model_name, model_anomalies in model_results.items():
            plt.figure(figsize=(14, 10))
            visualizer.plot_tsne_features(
                features, 
                model_anomalies,
                save_path=os.path.join(output_dirs['tsne'], f"tsne_{model_name}.png")
            )
            plt.close()
        
        # Plot behavioral features correlation
        print("\nGenerating behavioral features correlation plot...")
        plt.figure(figsize=(12, 10))
        visualizer.plot_behavioral_features_correlation(
            df,
            save_path=os.path.join(output_dirs['correlation'], "behavioral_features_correlation.png")
        )
        plt.close()
        
        # Plot rating distribution with enhanced styling
        print("\nGenerating rating distribution plot...")
        plt.figure(figsize=(12, 6))
        visualizer.plot_rating_distribution(df)
        plt.savefig(os.path.join(output_dirs['plots'], "rating_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot review length vs rating
        print("\nGenerating review length vs rating plot...")
        plt.figure(figsize=(12, 7))
        visualizer.plot_review_length_vs_rating(df)
        plt.savefig(os.path.join(output_dirs['plots'], "review_length_vs_rating.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot anomaly distribution with enhanced visualization
        print("\nGenerating anomaly distribution plot...")
        plt.figure(figsize=(12, 7))
        visualizer.plot_anomaly_distribution(df, anomalies)
        plt.savefig(os.path.join(output_dirs['plots'], "anomaly_distribution.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot DBN layer scores with improved styling
        print("\nGenerating DBN layer scores plot...")
        plt.figure(figsize=(12, 7))
        visualizer.plot_dbn_layer_scores(layer_scores)
        plt.savefig(os.path.join(output_dirs['plots'], "dbn_layer_scores.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot model comparison with enhanced visualization
        print("\nGenerating model comparison plot...")
        model_comp_path = os.path.join(output_dirs['plots'], "model_comparison.png")
        print(f"Saving model comparison plot to: {model_comp_path}")
        plt.figure(figsize=(12, 7))
        visualizer.plot_model_comparison(model_results)
        plt.savefig(model_comp_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot ROC curves for all models
        print("\nGenerating ROC curves comparison...")
        plt.figure(figsize=(10, 8))
        # Sample ROC curve data for demonstration
        dbn_fpr = np.linspace(0, 1, 100)
        dbn_tpr = np.linspace(0, 1, 100) ** 2
        other_models_curves = {
            'Isolation Forest': (np.linspace(0, 1, 100), np.linspace(0, 1, 100) ** 1.5),
            'LOF': (np.linspace(0, 1, 100), np.linspace(0, 1, 100) ** 1.8),
            'One-Class SVM': (np.linspace(0, 1, 100), np.linspace(0, 1, 100) ** 1.6)
        }
        visualizer.plot_dbn_roc_curves(dbn_fpr, dbn_tpr, other_models_curves)
        plt.savefig(os.path.join(output_dirs['plots'], "roc_curves_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Verify files were created
        print("\nVerifying generated files:")
        expected_files = [
            os.path.join(output_dirs['plots'], "rating_distribution.png"),
            os.path.join(output_dirs['plots'], "review_length_vs_rating.png"),
            os.path.join(output_dirs['plots'], "anomaly_distribution.png"),
            os.path.join(output_dirs['plots'], "dbn_layer_scores.png"),
            model_comp_path
        ]
        
        # Add t-SNE files
        for model_name in ['isolation_forest', 'lof', 'one_class_svm', 'hbos']:
            expected_files.append(os.path.join(output_dirs['tsne'], f"tsne_{model_name}.png"))
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path} exists")
            else:
                print(f"✗ {file_path} is missing")
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
        raise

def main():
    """Main function to demonstrate visualization capabilities."""
    print("Starting visualization demonstration...")
    
    # Create sample data
    print("\nGenerating sample data...")
    df, features, anomalies, model_results, layer_scores = create_sample_data(n_samples=1000)
    
    # Create output directories
    print("\nCreating output directories...")
    output_dirs = create_output_directories()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generate_visualizations(df, features, anomalies, model_results, layer_scores, output_dirs)
    
    print("\nVisualization demonstration completed!")

if __name__ == "__main__":
    main()