import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

class Visualizer:
    @staticmethod
    def plot_rating_distribution(df):
        """Plot distribution of review scores with improved styling."""
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df, x='review/score', bins=10, color='skyblue', edgecolor='black')
        plt.title('Distribution of Review Scores', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_helpfulness_vs_rating(df):
        """Plot helpfulness ratio vs rating with enhanced visualization."""
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='review/score', y='helpfulness_ratio', data=df, palette='viridis')
        plt.title('Helpfulness Ratio vs Rating', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Helpfulness Ratio', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_temporal_patterns(df):
        """Plot temporal patterns with improved time series visualization."""
        monthly_ratings = df.set_index('review_time')['review/score'].resample('M').agg(['mean', 'count'])
        
        # Create two y-axes
        fig, ax1 = plt.subplots(figsize=(15, 7))
        ax2 = ax1.twinx()
        
        # Plot average rating
        ax1.plot(monthly_ratings.index, monthly_ratings['mean'], 'b-', label='Average Rating')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Average Rating', color='b', fontsize=12)
        ax1.tick_params(axis='y', labelcolor='b')
        
        # Plot review count
        ax2.bar(monthly_ratings.index, monthly_ratings['count'], alpha=0.3, color='gray', label='Review Count')
        ax2.set_ylabel('Number of Reviews', color='gray', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='gray')
        
        plt.title('Temporal Patterns in Reviews', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_review_length_vs_rating(df):
        """Plot review length vs rating with enhanced visualization."""
        df['review_length'] = df['review/text'].str.len()
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='review/score', y='review_length', data=df, palette='viridis')
        plt.title('Review Length vs Rating', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Review Length (characters)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_anomaly_distribution(df, anomalies):
        """Plot anomaly distribution with enhanced visualization."""
        plt.figure(figsize=(12, 7))
        scatter = sns.scatterplot(
            data=df,
            x='review/score',
            y='helpfulness_ratio',
            hue=anomalies,
            palette=['blue', 'red'],
            alpha=0.6,
            s=100
        )
        plt.title('Anomaly Distribution in Reviews', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Helpfulness Ratio', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(title='Anomaly Status', labels=['Normal', 'Anomaly'])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_layer_weights(weights, layer_idx):
        """Plot DBN layer weights visualization."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(weights, cmap='viridis', center=0)
        plt.title(f'DBN Layer {layer_idx} Weight Distribution', fontsize=14, pad=20)
        plt.xlabel('Input Features', fontsize=12)
        plt.ylabel('Hidden Units', fontsize=12)
        plt.colorbar(label='Weight Value')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_reconstruction_error(error_distribution):
        """Plot DBN reconstruction error distribution."""
        plt.figure(figsize=(12, 6))
        sns.histplot(error_distribution, bins=50, kde=True)
        plt.title('DBN Reconstruction Error Distribution', fontsize=14, pad=20)
        plt.xlabel('Reconstruction Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_learning_progress(train_errors, val_errors=None):
        """Plot DBN learning progress over epochs."""
        plt.figure(figsize=(12, 6))
        plt.plot(train_errors, label='Training Error', color='blue')
        if val_errors is not None:
            plt.plot(val_errors, label='Validation Error', color='red', linestyle='--')
        plt.title('DBN Learning Progress', fontsize=14, pad=20)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Error', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_anomaly_threshold(anomaly_scores, threshold):
        """Plot DBN anomaly scores with threshold."""
        plt.figure(figsize=(12, 6))
        sns.histplot(anomaly_scores, bins=50, kde=True)
        plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.3f}')
        plt.title('DBN Anomaly Score Distribution', fontsize=14, pad=20)
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_precision_recall(y_true, y_scores):
        """Plot precision-recall curve for DBN model."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, 'b-', linewidth=2)
        plt.title('DBN Precision-Recall Curve', fontsize=14, pad=20)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_confusion_matrix(y_true, y_pred):
        """Plot confusion matrix for DBN predictions."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('DBN Confusion Matrix', fontsize=14, pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_feature_importance(feature_importance, feature_names):
        """Plot feature importance for DBN model."""
        plt.figure(figsize=(12, 6))
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.barh(pos, feature_importance[sorted_idx])
        plt.yticks(pos, feature_names[sorted_idx])
        plt.title('DBN Feature Importance', fontsize=14, pad=20)
        plt.xlabel('Importance Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_roc_curve(y_true, y_scores):
        """Plot ROC curve for DBN model."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('DBN ROC Curve', fontsize=14, pad=20)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_dbn_anomaly_scores_by_rating(df, anomaly_scores):
        """Plot anomaly scores distribution by rating."""
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='review/score', y=anomaly_scores, data=df)
        plt.title('Anomaly Scores Distribution by Rating', fontsize=14, pad=20)
        plt.xlabel('Rating', fontsize=12)
        plt.ylabel('Anomaly Score', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()