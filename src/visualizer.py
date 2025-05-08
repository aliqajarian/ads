import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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