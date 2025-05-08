import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import kagglehub
import os
import shutil
from pathlib import Path
import logging
from data_downloader import DataDownloader
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self):
        # Initialize DataDownloader
        self.downloader = DataDownloader(dataset_name="mohamedbakhet/amazon-books-reviews", download_dir="data/raw")
        # Get downloaded file paths
        downloaded_files = self.downloader.download_all()
        self.reviews_path = downloaded_files['reviews']
        self.books_path = downloaded_files['books_details']

    def load_data(self):
        print(f"Loading data from: {self.downloader.data_dir}")
        
        # Load the CSV files
        try:
            reviews_df = pd.read_csv(self.reviews_path)
            books_df = pd.read_csv(self.books_path)
            
            # Process helpfulness ratings
            reviews_df['helpfulness_numerator'] = reviews_df['review/helpfulness'].str.split('/').str[0].astype(float)
            reviews_df['helpfulness_denominator'] = reviews_df['review/helpfulness'].str.split('/').str[1].astype(float)
            reviews_df['helpfulness_ratio'] = reviews_df['helpfulness_numerator'] / reviews_df['helpfulness_denominator']
            reviews_df['User_id'] = reviews_df['User_id']
            
            # Convert review time to datetime
            reviews_df['review_time'] = pd.to_datetime(reviews_df['review/time'], unit='s')
            
            # Merge the dataframes
            merged_df = pd.merge(reviews_df, books_df, on='Title', how='inner')
            
            # Preprocess text data
            merged_df['review/text'] = merged_df['review/text'].fillna('')
            merged_df['review/summary'] = merged_df['review/summary'].fillna('')
            merged_df['categories'] = merged_df['categories'].fillna('')
            
            # Impute missing numerical values
            imputer = SimpleImputer(strategy='mean')
            numerical_columns = ['review/score', 'helpfulness_ratio', 'ratingsCount']
            merged_df[numerical_columns] = imputer.fit_transform(merged_df[numerical_columns])
            
            # Drop rows with NaN values in numerical columns
            merged_df.dropna(subset=numerical_columns, inplace=True)
            
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
        user_review_counts = df['User_id'].map(df['User_id'].value_counts()).values.reshape(-1, 1)
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
        
        # Impute any remaining NaNs
        imputer = SimpleImputer(strategy='mean')
        combined_features = imputer.fit_transform(combined_features)
        
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