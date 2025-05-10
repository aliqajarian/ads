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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    DataLoader class for loading and preprocessing Amazon book review data.
    
    This class handles downloading, loading, preprocessing, and feature engineering
    for the Amazon book reviews dataset. It supports loading either the full dataset
    or a limited subset (e.g., 500k records) for faster processing during development
    and testing.
    """
    def __init__(self):
        """
        Initialize the DataLoader with data paths.
        
        Downloads the required dataset files and sets up file paths for processing.
        """
        # Initialize DataDownloader
        self.downloader = DataDownloader(data_dir="data/raw")
        # Get downloaded file paths
        try:
            downloaded_files = self.downloader.download_all()
            
            # Print available keys for debugging
            print("Available file keys:", list(downloaded_files.keys()))
            
            # Use file stem names instead of hardcoded keys
            # 'Books_rating' instead of 'reviews'
            # 'books_data' instead of 'books_details'
            self.reviews_path = downloaded_files['Books_rating']
            self.books_path = downloaded_files['books_data']
            
            # Verify files exist
            if not os.path.exists(self.reviews_path):
                raise FileNotFoundError(f"Reviews file not found at: {self.reviews_path}")
            if not os.path.exists(self.books_path):
                raise FileNotFoundError(f"Books details file not found at: {self.books_path}")
                
        except Exception as e:
            logger.error(f"Error initializing DataLoader: {str(e)}")
            raise

    def load_data(self, max_records=500000, use_full_dataset=False):
        """
        Load and preprocess the dataset.
        
        Args:
            max_records (int): Maximum number of records to load (default: 500,000)
            use_full_dataset (bool): If True, ignore max_records and use the full dataset
            
        Returns:
            pd.DataFrame: Processed and merged dataset
        """
        logger.info(f"Loading data from: {self.downloader.data_dir}")
        
        try:
            # Load the CSV files
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
            
            # Limit the dataset size if requested
            total_records = len(merged_df)
            if not use_full_dataset and total_records > max_records:
                logger.info(f"Limiting dataset from {total_records} to {max_records} records")
                # Use stratified sampling based on review scores to maintain distribution
                merged_df = merged_df.groupby('review/score', group_keys=False).apply(
                    lambda x: x.sample(min(len(x), int(max_records * len(x) / total_records)), random_state=42)
                )
                
                # If we still have more than max_records (due to rounding), take exactly max_records
                if len(merged_df) > max_records:
                    merged_df = merged_df.sample(max_records, random_state=42)
            
            logger.info(f"Successfully loaded {len(merged_df)} reviews")
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def prepare_features(self, df):
        """
        Extract and engineer features from the dataset.
        
        This method works with both the full dataset and the limited dataset
        (when max_records parameter is used in load_data). The feature extraction
        process is the same regardless of dataset size.
        
        Args:
            df: DataFrame containing the preprocessed data
            
        Returns:
            numpy.ndarray: Combined feature matrix
        """
        try:
            logger.info(f"Preparing features for {len(df)} records...")
            
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
            
            logger.info(f"Feature matrix shape: {combined_features.shape}")
            return combined_features
                
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise
    
    def split_train_test(self, features, df, test_size=0.2, random_state=42):
        """
        Splits features and DataFrame into train and test sets.
        
        This method works with both the full dataset and the limited dataset (when max_records
        parameter is used in load_data). The stratification in load_data ensures that the
        distribution of review scores is maintained even when using a subset of the data.
        
        Args:
            features: Feature matrix from prepare_features method
            df: DataFrame containing the original data
            test_size: Proportion of the dataset to include in the test split (default: 0.2)
            random_state: Random seed for reproducibility (default: 42)
            
        Returns: 
            X_train, X_test, df_train, df_test: Train and test splits for both features and DataFrame
        """
        try:
            X_train, X_test, df_train, df_test = train_test_split(
                features, df, test_size=test_size, random_state=random_state, shuffle=True
            )
            logger.info(f"Split data into training set ({len(X_train)} samples) and test set ({len(X_test)} samples)")
            return X_train, X_test, df_train, df_test
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise