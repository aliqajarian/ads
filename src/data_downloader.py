import os
import kagglehub

class DataDownloader:
    def __init__(self, dataset_name, download_dir):
        self.dataset_name = dataset_name
        self.download_dir = download_dir

    def download_all(self):
        # Ensure the download directory exists
        os.makedirs(self.download_dir, exist_ok=True)

        # Use kagglehub to download the dataset
        kagglehub.dataset(self.dataset_name).download(self.download_dir)

        # Return paths to the downloaded files
        return {
            'reviews': os.path.join(self.download_dir, 'reviews.csv'),
            'books_details': os.path.join(self.download_dir, 'books_details.csv')
        }