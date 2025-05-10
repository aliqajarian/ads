import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pyod.models.hbos import HBOS
from sklearn.cluster import DBSCAN

class Visualizer:
    def plot_rating_distribution(self, df):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='review/score', data=df)
        plt.title('Distribution of Review Scores')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()

    def plot_review_length_vs_rating(self, df):
        df['review_length'] = df['review/text'].apply(len)
        plt.figure(figsize=(12, 7))
        sns.boxplot(x='review/score', y='review_length', data=df)
        plt.title('Review Length vs. Rating')
        plt.xlabel('Rating')
        plt.ylabel('Review Length (characters)')
        plt.ylim(0, df['review_length'].quantile(0.95)) # Limiting y-axis for better visualization
        plt.show()

    def plot_anomaly_distribution(self, df, anomalies):
        df_copy = df.copy()
        df_copy['anomaly'] = anomalies
        plt.figure(figsize=(10, 6))
        sns.countplot(x='review/score', hue='anomaly', data=df_copy)
        plt.title('Distribution of Anomalies by Review Score')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.legend(title='Is Anomaly?')
        plt.show()

    def plot_dbn_layer_scores(self, layer_scores):
        """
        Plots the training and validation scores (pseudo-likelihood) for each RBM layer in the DBN.

        Parameters:
        - layer_scores (list of dict): A list where each dictionary contains 'train_score'
                                       and optionally 'val_score' for an RBM layer.
        """
        num_layers = len(layer_scores)
        train_scores = [score['train_score'] for score in layer_scores]
        val_scores = [score.get('val_score') for score in layer_scores]
        # Filter out None values for validation scores if some layers didn't have them
        valid_val_scores_indices = [i for i, score in enumerate(val_scores) if score is not None]
        filtered_val_scores = [val_scores[i] for i in valid_val_scores_indices]

        layers = np.arange(1, num_layers + 1)

        plt.figure(figsize=(12, 7))
        plt.plot(layers, train_scores, marker='o', linestyle='-', label='Training Score (Pseudo-Likelihood)')
        
        if filtered_val_scores and len(filtered_val_scores) == len(valid_val_scores_indices):
             # Ensure we only plot validation scores if they exist and match the number of layers they correspond to
            plt.plot(np.array(layers)[valid_val_scores_indices], filtered_val_scores, marker='x', linestyle='--', label='Validation Score (Pseudo-Likelihood)')
        
        plt.title('DBN RBM Layer-wise Scores')
        plt.xlabel('RBM Layer Number')
        plt.ylabel('Average Pseudo-Likelihood')
        plt.xticks(layers)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_model_comparison(self, all_anomalies_results):
        """
        Plots a comparison of the number of anomalies detected by different models.

        Parameters:
        - all_anomalies_results (dict): A dictionary where keys are model names (str)
                                          and values are the anomaly predictions (array-like of booleans or 0/1).
        """
        model_names = list(all_anomalies_results.keys())
        anomaly_counts = [sum(results) for results in all_anomalies_results.values()]

        plt.figure(figsize=(12, 7))
        sns.barplot(x=model_names, y=anomaly_counts)
        plt.title('Comparison of Anomalies Detected by Different Models')
        plt.xlabel('Model')
        plt.ylabel('Number of Anomalies Detected')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_tsne_features(self, features, anomalies, save_path=None):
        """
        Create t-SNE visualization of DBN-extracted features.
        
        Args:
            features: DBN-transformed features
            anomalies: Boolean array indicating anomalies
            save_path: Optional path to save the plot
        """
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            features_tsne[:, 0], 
            features_tsne[:, 1], 
            c=anomalies, 
            cmap='coolwarm', 
            alpha=0.6,
            s=50
        )
        
        plt.title("t-SNE Visualization of DBN Features", fontsize=14)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        
        # Add colorbar with custom labels
        cbar = plt.colorbar(scatter)
        cbar.set_label('Anomaly Status', fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Normal', 'Anomaly'])
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend(
            handles=scatter.legend_elements()[0],
            labels=['Normal', 'Anomaly'],
            title='Review Type',
            loc='upper right'
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"t-SNE plot saved to {save_path}")
        
        plt.show()

    def plot_behavioral_features_correlation(self, df, save_path=None):
        """
        Create correlation heatmap of behavioral features.
        
        Args:
            df: DataFrame containing the behavioral features
            save_path: Optional path to save the plot
        """
        # Prepare behavioral features
        behavioral_features = pd.DataFrame({
            'Review Length': df['review/text'].str.len(),
            'Rating': df['review/score'],
            'Helpfulness Ratio': df['helpfulness_ratio'],
            'User Activity': df['User_id'].map(df['User_id'].value_counts()),
            'Days Since First Review': (df['review_time'] - df['review_time'].min()).dt.days,
            'Review Time (Unix)': df['review_time'].astype(np.int64) // 10**9
        })
        
        # Calculate correlation matrix
        corr_matrix = behavioral_features.corr()
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            cmap='viridis',
            center=0,
            fmt='.2f',
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .8}
        )
        
        plt.title("Correlation Heatmap of Behavioral Features", fontsize=14, pad=20)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        
        plt.show()

    def plot_learning_curves(self, X, y, save_path=None):
        """
        Plot learning curves for all models to analyze their sensitivity to training data size.
        
        Args:
            X: Feature matrix
            y: Target labels
            save_path: Optional directory path to save the plots
        """
        # Define models to analyze
        models = {
            'Isolation Forest': IsolationForest(random_state=42),
            'Local Outlier Factor': LocalOutlierFactor(novelty=True),
            'One-Class SVM': OneClassSVM(),
            'HBOS': HBOS(),
            'DBSCAN': DBSCAN()
        }
        
        # Create a figure with subplots for each model
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()
        
        for idx, (model_name, model) in enumerate(models.items()):
            if idx >= len(axes):
                break
                
            # Calculate learning curve
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y,
                cv=5,
                scoring='f1',
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=-1
            )
            
            # Calculate mean and std for train and test scores
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            test_std = np.std(test_scores, axis=1)
            
            # Plot learning curve
            ax = axes[idx]
            ax.plot(train_sizes, train_mean, label='Train F1', color='blue')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            ax.plot(train_sizes, test_mean, label='Test F1', color='red')
            ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
            
            # Customize plot
            ax.set_title(f'Learning Curve - {model_name}')
            ax.set_xlabel('Training Size')
            ax.set_ylabel('F1 Score')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='lower right')
            
            # Set y-axis limits to be consistent across plots
            ax.set_ylim([0, 1.1])
        
        # Remove empty subplot if any
        if len(models) < len(axes):
            fig.delaxes(axes[-1])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves plot saved to {save_path}")
        
        plt.show()