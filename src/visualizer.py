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
from sklearn.metrics import f1_score

class Visualizer:
    def plot_rating_distribution(self, df):
        """Plot distribution of review scores."""
        try:
            print("Generating rating distribution plot...")
            plt.figure(figsize=(10, 6))
            sns.countplot(x='review/score', data=df)
            plt.title('Distribution of Review Scores')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.show()
            print("Rating distribution plot generated successfully")
        except Exception as e:
            print(f"Error generating rating distribution plot: {str(e)}")

    def plot_review_length_vs_rating(self, df):
        """Plot review length vs rating."""
        try:
            print("Generating review length vs rating plot...")
            df['review_length'] = df['review/text'].apply(len)
            plt.figure(figsize=(12, 7))
            sns.boxplot(x='review/score', y='review_length', data=df)
            plt.title('Review Length vs. Rating')
            plt.xlabel('Rating')
            plt.ylabel('Review Length (characters)')
            plt.ylim(0, df['review_length'].quantile(0.95))
            plt.show()
            print("Review length vs rating plot generated successfully")
        except Exception as e:
            print(f"Error generating review length vs rating plot: {str(e)}")

    def plot_anomaly_distribution(self, df, anomalies):
        """Plot distribution of anomalies by review score."""
        try:
            print("Generating anomaly distribution plot...")
            df_copy = df.copy()
            df_copy['anomaly'] = anomalies
            plt.figure(figsize=(10, 6))
            sns.countplot(x='review/score', hue='anomaly', data=df_copy)
            plt.title('Distribution of Anomalies by Review Score')
            plt.xlabel('Rating')
            plt.ylabel('Count')
            plt.legend(title='Is Anomaly?')
            plt.show()
            print("Anomaly distribution plot generated successfully")
        except Exception as e:
            print(f"Error generating anomaly distribution plot: {str(e)}")

    def plot_dbn_layer_scores(self, layer_scores):
        """Plot DBN layer scores."""
        try:
            print("Generating DBN layer scores plot...")
            if not layer_scores:
                print("Warning: No layer scores provided for DBN plot")
                return

            num_layers = len(layer_scores)
            train_scores = [score['train_score'] for score in layer_scores]
            val_scores = [score.get('val_score') for score in layer_scores]
            valid_val_scores_indices = [i for i, score in enumerate(val_scores) if score is not None]
            filtered_val_scores = [val_scores[i] for i in valid_val_scores_indices]

            layers = np.arange(1, num_layers + 1)

            plt.figure(figsize=(12, 7))
            plt.plot(layers, train_scores, marker='o', linestyle='-', label='Training Score (Pseudo-Likelihood)')
            
            if filtered_val_scores and len(filtered_val_scores) == len(valid_val_scores_indices):
                plt.plot(np.array(layers)[valid_val_scores_indices], filtered_val_scores, 
                        marker='x', linestyle='--', label='Validation Score (Pseudo-Likelihood)')
            
            plt.title('DBN RBM Layer-wise Scores')
            plt.xlabel('RBM Layer Number')
            plt.ylabel('Average Pseudo-Likelihood')
            plt.xticks(layers)
            plt.legend()
            plt.grid(True)
            plt.show()
            print("DBN layer scores plot generated successfully")
        except Exception as e:
            print(f"Error generating DBN layer scores plot: {str(e)}")

    def plot_model_comparison(self, all_anomalies_results):
        """Plot model comparison."""
        try:
            print("Generating model comparison plot...")
            if not all_anomalies_results:
                print("Warning: No anomaly results provided for model comparison plot")
                return

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
            print("Model comparison plot generated successfully")
        except Exception as e:
            print(f"Error generating model comparison plot: {str(e)}")

    def plot_tsne_features(self, features, anomalies, save_path=None):
        """Plot t-SNE visualization."""
        try:
            print("Generating t-SNE visualization...")
            if features is None or anomalies is None:
                print("Warning: Missing features or anomalies data for t-SNE plot")
                return

            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42)
            features_tsne = tsne.fit_transform(features)
            
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
            
            cbar = plt.colorbar(scatter)
            cbar.set_label('Anomaly Status', fontsize=12)
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Normal', 'Anomaly'])
            
            plt.grid(True, linestyle='--', alpha=0.7)
            
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
            print("t-SNE visualization generated successfully")
        except Exception as e:
            print(f"Error generating t-SNE visualization: {str(e)}")

    def plot_behavioral_features_correlation(self, df, save_path=None):
        """Plot behavioral features correlation."""
        try:
            print("Generating behavioral features correlation plot...")
            if df is None:
                print("Warning: No data provided for behavioral features correlation plot")
                return

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
            
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
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
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Correlation heatmap saved to {save_path}")
            
            plt.show()
            print("Behavioral features correlation plot generated successfully")
        except Exception as e:
            print(f"Error generating behavioral features correlation plot: {str(e)}")

    def plot_learning_curves(self, X, y, save_path=None):
        """Plot learning curves."""
        try:
            print("Generating learning curves...")
            if X is None or y is None:
                print("Warning: Missing features or labels for learning curves plot")
                return

            # Define models to analyze with simplified parameters
            models = {
                'Isolation Forest': IsolationForest(
                    n_estimators=100,
                    random_state=42
                ),
                'Local Outlier Factor': LocalOutlierFactor(
                    n_neighbors=20,
                    novelty=True
                ),
                'One-Class SVM': OneClassSVM(
                    kernel='rbf',
                    nu=0.1
                ),
                'HBOS': HBOS(
                    n_bins=10,
                    alpha=0.1
                ),
                'DBSCAN': DBSCAN(
                    eps=0.5,
                    min_samples=5
                )
            }
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            axes = axes.ravel()
            
            for idx, (model_name, model) in enumerate(models.items()):
                if idx >= len(axes):
                    break
                    
                print(f"\nCalculating learning curve for {model_name}...")
                
                def custom_f1_scorer(estimator, X, y):
                    try:
                        if hasattr(estimator, 'fit_predict'):
                            y_pred = estimator.fit_predict(X)
                        else:
                            estimator.fit(X)
                            y_pred = estimator.predict(X)
                        y_pred_binary = np.where(y_pred == -1, 1, 0)
                        return f1_score(y, y_pred_binary, average='weighted', zero_division=1)
                    except Exception as e:
                        print(f"Error in scoring for {model_name}: {str(e)}")
                        return 0.0
                
                try:
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X, y,
                        cv=3,
                        scoring=custom_f1_scorer,
                        train_sizes=np.linspace(0.2, 1.0, 5),
                        n_jobs=1,
                        verbose=1
                    )
                    
                    train_mean = np.mean(train_scores, axis=1)
                    train_std = np.std(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)
                    test_std = np.std(test_scores, axis=1)
                    
                    ax = axes[idx]
                    ax.plot(train_sizes, train_mean, label='Train F1', color='blue')
                    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
                    ax.plot(train_sizes, test_mean, label='Test F1', color='red')
                    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
                    
                    ax.set_title(f'Learning Curve - {model_name}')
                    ax.set_xlabel('Training Size')
                    ax.set_ylabel('F1 Score')
                    ax.grid(True, linestyle='--', alpha=0.7)
                    ax.legend(loc='lower right')
                    ax.set_ylim([0, 1.1])
                    
                except Exception as e:
                    print(f"Error calculating learning curve for {model_name}: {str(e)}")
                    ax = axes[idx]
                    ax.text(0.5, 0.5, f'Error: {str(e)}', 
                           horizontalalignment='center',
                           verticalalignment='center',
                           transform=ax.transAxes)
                    ax.set_title(f'Learning Curve - {model_name}')
            
            if len(models) < len(axes):
                fig.delaxes(axes[-1])
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Learning curves plot saved to {save_path}")
            
            plt.show()
            print("Learning curves plot generated successfully")
        except Exception as e:
            print(f"Error generating learning curves plot: {str(e)}")