import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_rating_distribution(df):
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='review/score', bins=10)
        plt.title('Rating Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Count')
        plt.show()
        
    @staticmethod
    def plot_helpfulness_vs_rating(df):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='review/score', y='helpfulness_ratio', data=df)
        plt.title('Helpfulness Ratio vs Rating')
        plt.xlabel('Rating')
        plt.ylabel('Helpfulness Ratio')
        plt.show()
        
    @staticmethod
    def plot_temporal_patterns(df):
        plt.figure(figsize=(12, 6))
        df.set_index('review_time')['review/score'].resample('M').mean().plot()
        plt.title('Average Rating Over Time')
        plt.xlabel('Time')
        plt.ylabel('Average Rating')
        plt.show()
    
    @staticmethod
    def plot_review_length_vs_rating(df):
        df['review_length'] = df['review/text'].str.len()
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='review/score', y='review_length', data=df)
        plt.title('Review Length vs Rating')
        plt.xlabel('Rating')
        plt.ylabel('Review Length')
        plt.show()
    
    @staticmethod
    def plot_anomaly_distribution(df, anomalies):
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            data=df,
            x='review/score',
            y='helpfulness_ratio',
            hue=anomalies,
            palette=['blue', 'red']
        )
        plt.title('Anomaly Distribution')
        plt.xlabel('Rating')
        plt.ylabel('Helpfulness Ratio')
        plt.show()