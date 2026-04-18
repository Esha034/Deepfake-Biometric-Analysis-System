import os
import json
import zipfile

def setup_kaggle():
    # Load kaggle credentials from the root project folder
    with open('kaggle.json', 'r') as f:
        kaggle_data = json.load(f)
    
    # Set environment variables for Kaggle API
    os.environ['KAGGLE_USERNAME'] = kaggle_data['username']
    os.environ['KAGGLE_KEY'] = kaggle_data['key']

def download_dataset():
    dataset_name = "ciplab/real-and-fake-face-detection"
    download_path = "data"
    
    if not os.path.exists(download_path):
        os.makedirs(download_path)
    
    print(f"Downloading dataset {dataset_name}...")
    # Use kaggle library if installed, otherwise recommend pip install
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_name, path=download_path, unzip=True)
        print("Dataset downloaded and extracted successfully.")
    except ImportError:
        print("Kaggle library not found. Falling back to command line...")
        os.system(f"kaggle datasets download -d {dataset_name} -p {download_path} --unzip")

if __name__ == "__main__":
    setup_kaggle()
    download_dataset()
