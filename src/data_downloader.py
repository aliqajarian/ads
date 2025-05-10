import kagglehub
import shutil
from pathlib import Path
import logging
from tqdm import tqdm
import os
import json
import pandas as pd

class DataDownloader:
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the data downloader.
        
        Args:
            data_dir (str): Directory to save downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Kaggle dataset ID
        self.dataset_id = "mohamedbakhet/amazon-books-reviews"
    
    def download_dataset(self) -> str:
        """
        Download the Kaggle dataset.
        
        Returns:
            str: Path to the downloaded dataset directory
        """
        try:
            # Download dataset from Kaggle
            self.logger.info(f"Downloading dataset: {self.dataset_id}")
            dataset_path = kagglehub.dataset_download(self.dataset_id)
            self.logger.info(f"Dataset downloaded to: {dataset_path}")
            
            # Print the contents of the dataset directory
            dataset_path = Path(dataset_path)
            self.logger.info("Dataset contents:")
            for file in dataset_path.glob("**/*"):
                if file.is_file():
                    self.logger.info(f"Found file: {file}")
            
            # Copy files to our data directory
            for file in dataset_path.glob("**/*"):
                if file.is_file():
                    dest_path = self.data_dir / file.name
                    shutil.copy2(file, dest_path)
                    self.logger.info(f"Copied {file.name} to {dest_path}")
            
            return str(dataset_path)
            
        except Exception as e:
            self.logger.error(f"Failed to download dataset: {str(e)}")
            raise
    
    def download_all(self) -> dict:
        """
        Download all required data files.
        
        Returns:
            dict: Dictionary of downloaded file paths
        """
        downloaded_files = {}
        
        try:
            # Download dataset
            dataset_path = self.download_dataset()
            downloaded_files['dataset'] = dataset_path
            
            # List all files in the data directory
            for file in self.data_dir.glob("*"):
                if file.is_file():
                    downloaded_files[file.stem] = str(file)
            
            return downloaded_files
            
            except Exception as e:
            self.logger.error(f"Failed to download data: {str(e)}")
            raise

def main():
    """Main function to download data."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize downloader
    downloader = DataDownloader()
    
    try:
        # Download all files
        downloaded_files = downloader.download_all()
        
        # Print downloaded file paths
        print("\nDownloaded files:")
        for file_type, path in downloaded_files.items():
            print(f"{file_type}: {path}")
            
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        raise

if __name__ == "__main__":
    main()