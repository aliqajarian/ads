import os
import kagglehub

class DataDownloader:
    def __init__(self, dataset_name, download_dir):
        self.dataset_name = dataset_name
        self.download_dir = download_dir
        # Colab-specific Kaggle input path
        self.colab_dataset_path = "/kaggle/input/amazon-books-reviews"

    @property
    def data_dir(self):
        # Always use Colab dataset path
        return self.colab_dataset_path

    def download_all(self):
        # If running in Colab and dataset is already present, use it directly
        if os.path.exists(self.colab_dataset_path):
            reviews_path = os.path.join(self.colab_dataset_path, 'Books_rating.csv')
            books_details_path = os.path.join(self.colab_dataset_path, 'books_data.csv')
            return {
                'reviews': reviews_path,
                'books_details': books_details_path
            }
        else:
            # Ensure the download directory exists
            os.makedirs(self.download_dir, exist_ok=True)
            # Use kagglehub to download the dataset - Skip download in Colab
            reviews_path = os.path.join(self.colab_dataset_path, 'Books_rating.csv')
            books_details_path = os.path.join(self.colab_dataset_path, 'books_data.csv')
        return {
            'reviews': reviews_path,
            'books_details': books_details_path
        }