import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import os

class DataLoader:
    def __init__(self):
        # Download dataset from Kaggle
        self.dataset_path = kagglehub.dataset_download("mohamedbakhet/amazon-books-reviews")
        
        # Define paths for the CSV files within the downloaded dataset
        self.reviews_path = os.path.join(self.dataset_path, "reviews.csv")
        self.books_path = os.path.join(self.dataset_path, "books_details.csv")
    
    def load_data(self):
        print(f"Loading data from: {self.dataset_path}")
        
        # Load the CSV files
        try:
            reviews_df = pd.read_csv(self.reviews_path)
            books_df = pd.read_csv(self.books_path)
            
            # Process helpfulness ratings
            reviews_df['helpfulness_numerator'] = reviews_df['review/helpfulness'].str.split('/').str[0].astype(float)
            reviews_df['helpfulness_denominator'] = reviews_df['review/helpfulness'].str.split('/').str[1].astype(float)
            reviews_df['helpfulness_ratio'] = reviews_df['helpfulness_numerator'] / reviews_df['helpfulness_denominator']
            
            # Convert review time to datetime
            reviews_df['review_time'] = pd.to_datetime(reviews_df['review/time'], unit='s')
            
            # Merge the dataframes
            merged_df = pd.merge(reviews_df, books_df, on='Title', how='inner')
            
            # Preprocess text data
            merged_df['review/text'] = merged_df['review/text'].fillna('')
            merged_df['review/summary'] = merged_df['review/summary'].fillna('')
            merged_df['categories'] = merged_df['categories'].fillna('')
            
            print(f"Successfully loaded {len(merged_df)} reviews")
            return merged_df
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            raise
    
    def prepare_features(self, df):
        # Extract text features from both review text and summary
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer_text = TfidfVectorizer(max_features=800)
        vectorizer_summary = TfidfVectorizer(max_features=200)
        
        text_features = vectorizer_text.fit_transform(df['review/text'])
        summary_features = vectorizer_summary.fit_transform(df['review/summary'])
        
        # Multi-factor feature engineering
        # Review length
        review_length = df['review/text'].str.len().values.reshape(-1, 1)
        # User activity: number of reviews per user
        user_review_counts = df['reviewerID'].map(df['reviewerID'].value_counts()).values.reshape(-1, 1)
        # Temporal pattern: days since earliest review
        min_time = df['review_time'].min()
        days_since_first = (df['review_time'] - min_time).dt.days.values.reshape(-1, 1)
        
        # Create numerical features
        numerical_features = np.column_stack([
            df['review/score'],
            df['helpfulness_ratio'],
            df['ratingsCount'],
            df['review_time'].astype(np.int64) // 10**9,  # Convert to Unix timestamp
            review_length,
            user_review_counts,
            days_since_first
        ])
        
        # Combine all features
        combined_features = np.hstack([
            text_features.toarray(),
            summary_features.toarray(),
            numerical_features
        ])
        
        return combined_features
    
    def split_train_test(self, features, df, test_size=0.2, random_state=42):
        """
        Splits features and DataFrame into train and test sets.
        Returns: X_train, X_test, df_train, df_test
        """
        X_train, X_test, df_train, df_test = train_test_split(
            features, df, test_size=test_size, random_state=random_state, shuffle=True
        )
        return X_train, X_test, df_train, df_test