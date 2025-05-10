import logging
import numpy as np
from data_loader import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Example script demonstrating the use of limited dataset size."""
    try:
        # Initialize the data loader
        logger.info("Initializing DataLoader...")
        data_loader = DataLoader()
        
        # Example 1: Load limited dataset (500k records)
        logger.info("\nExample 1: Loading limited dataset (500k records)")
        df_limited = data_loader.load_data(max_records=500000, use_full_dataset=False)
        logger.info(f"Limited dataset size: {len(df_limited)} records")
        
        # Prepare features for limited dataset
        logger.info("Preparing features for limited dataset...")
        features_limited = data_loader.prepare_features(df_limited)
        logger.info(f"Feature matrix shape: {features_limited.shape}")
        
        # Split limited dataset
        logger.info("Splitting limited dataset into train and test sets...")
        X_train_limited, X_test_limited, df_train_limited, df_test_limited = \
            data_loader.split_train_test(features_limited, df_limited)
        
        logger.info(f"Training set size: {len(df_train_limited)} records")
        logger.info(f"Test set size: {len(df_test_limited)} records")
        
        # Example 2: Load full dataset (if needed for comparison)
        logger.info("\nExample 2: Loading full dataset (optional)")
        logger.info("Note: This will use the full dataset, which may be very large")
        logger.info("Uncomment the following code to run with the full dataset")
        
        '''
        df_full = data_loader.load_data(use_full_dataset=True)
        logger.info(f"Full dataset size: {len(df_full)} records")
        
        # Prepare features for full dataset
        logger.info("Preparing features for full dataset...")
        features_full = data_loader.prepare_features(df_full)
        logger.info(f"Feature matrix shape: {features_full.shape}")
        
        # Split full dataset
        logger.info("Splitting full dataset into train and test sets...")
        X_train_full, X_test_full, df_train_full, df_test_full = \
            data_loader.split_train_test(features_full, df_full)
        
        logger.info(f"Training set size: {len(df_train_full)} records")
        logger.info(f"Test set size: {len(df_test_full)} records")
        '''
        
        logger.info("\nDataset comparison complete!")
        
    except Exception as e:
        logger.error(f"Error in example script: {str(e)}")
        raise

if __name__ == "__main__":
    main()