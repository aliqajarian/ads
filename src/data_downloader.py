import os
import kagglehub
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataDownloader:
    def __init__(self, dataset_name, download_dir):
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        try:
            os.makedirs(download_dir, exist_ok=True)
            logger.info(f"Initialized DataDownloader with dataset: {dataset_name}")
        except Exception as e:
            logger.error(f"Error creating download directory: {str(e)}")
            raise

    def download_all(self):
        try:
            logger.info(f"Downloading dataset {self.dataset_name} to {self.download_dir}")
            # Download the dataset using kagglehub
            kagglehub.model_download(self.dataset_name, path=self.download_dir)
            
            # Define expected file paths
            reviews_path = os.path.join(self.download_dir, 'Books_rating.csv')
            books_details_path = os.path.join(self.download_dir, 'Books_details.csv')
            
            # Verify files were downloaded
            if not os.path.exists(reviews_path):
                raise FileNotFoundError(f"Reviews file not found at: {reviews_path}")
            if not os.path.exists(books_details_path):
                raise FileNotFoundError(f"Books details file not found at: {books_details_path}")
            
            logger.info("Dataset downloaded successfully")
            return {
                'reviews': reviews_path,
                'books_details': books_details_path
            }
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise