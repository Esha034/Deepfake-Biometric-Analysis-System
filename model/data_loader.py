import os
import shutil
import random
from tqdm import tqdm
import config

def clear_processed_dir():
    if os.path.exists(config.PROCESSED_DIR):
        print(f"Clearing existing processed data in {config.PROCESSED_DIR}...")
        shutil.rmtree(config.PROCESSED_DIR)
    os.makedirs(config.TRAIN_DIR)
    os.makedirs(config.VAL_DIR)
    os.makedirs(config.TEST_DIR)

def split_data(train_split=0.7, val_split=0.15, test_split=0.15):
    # Setup folders
    for split in ['train', 'valid', 'test']:
        for label in ['real', 'fake']:
            os.makedirs(os.path.join(config.PROCESSED_DIR, split, label), exist_ok=True)

    # Path to source folders
    real_src = os.path.join(config.RAW_DATA_PATH, 'training_real')
    fake_src = os.path.join(config.RAW_DATA_PATH, 'training_fake')

    if not os.path.exists(real_src):
        # Fallback check for alternate folder structures if extraction naming varies
        real_src = os.path.join(config.DATA_DIR, 'real_and_fake_face', 'training_real')
        fake_src = os.path.join(config.DATA_DIR, 'real_and_fake_face', 'training_fake')

    print(f"Sourcing data from: {real_src} and {fake_src}")

    # Process REAL images
    real_files = [f for f in os.listdir(real_src) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(real_files)
    
    # Process FAKE images - Group by difficulty
    fake_files = [f for f in os.listdir(fake_src) if f.endswith(('.jpg', '.jpeg', '.png'))]
    difficulties = {'easy': [], 'mid': [], 'hard': []}
    
    for f in fake_files:
        if 'easy' in f.lower(): difficulties['easy'].append(f)
        elif 'mid' in f.lower(): difficulties['mid'].append(f)
        elif 'hard' in f.lower(): difficulties['hard'].append(f)
        else: difficulties['easy'].append(f) # Default to easy if unknown

    print(f"Total Real: {len(real_files)}")
    print(f"Total Fake Breakdown: Easy: {len(difficulties['easy'])} | Mid: {len(difficulties['mid'])} | Hard: {len(difficulties['hard'])}")

    # Function to distribute files
    def distribute(files, label):
        n = len(files)
        tr = int(n * train_split)
        vl = int(n * val_split)
        
        splits = {
            'train': files[:tr],
            'valid': files[tr:tr+vl],
            'test':  files[tr+vl:]
        }
        
        src_path = real_src if label == 'real' else fake_src
        for split_name, split_files in splits.items():
            dest = os.path.join(config.PROCESSED_DIR, split_name, label)
            for f in split_files:
                shutil.copy(os.path.join(src_path, f), os.path.join(dest, f))

    # Distribute Reals
    distribute(real_files, 'real')
    
    # Distribute Fakes per difficulty to ensure balanced representative in each set
    for diff in difficulties:
        files = difficulties[diff]
        random.shuffle(files)
        distribute(files, 'fake')

    print("Data balancing and splitting complete.")

if __name__ == "__main__":
    split_data()
