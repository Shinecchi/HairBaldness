import os
import pandas as pd
import shutil
import requests
import zipfile
import io

# --- CONFIGURATION ---
BASE_DIR = os.path.join(os.getcwd(), "dataset_v1_new")
DOWNLOAD_URL = "https://app.roboflow.com/ds/1vWfI00bU2?key=Fp6Y8x19ee"

# --- UPDATED LABEL MAPPING ---
label_map = {
    'LEVEL_1': 'healthy',           # Level 1 is now the standard "Healthy"
    'LEVEL_2': 'early_sign',        # Level 2 is specifically "Early Sign"
    'LEVEL_3': 'mild_thinning',
    'LEVEL_4': 'moderate_thinning',
    'LEVEL_5': 'severe_thinning',
    'LEVEL_6': 'balding',
    'LEVEL_7': 'bald'
}

def download_data_if_needed():
    # Check if the data exists
    train_csv = os.path.join(BASE_DIR, "train", "_classes.csv")

    if os.path.exists(train_csv):
        print(f"--- Data already found at {BASE_DIR}. Skipping download. ---")
        return

    print(f"--- Data missing. Downloading from Roboflow... ---")
    try:
        response = requests.get(DOWNLOAD_URL)
        if response.status_code == 200:
            print("Download complete. Extracting...")
            z = zipfile.ZipFile(io.BytesIO(response.content))
            z.extractall(BASE_DIR)
            print(f"Success! Data extracted to: {BASE_DIR}")
        else:
            print(f"CRITICAL ERROR: Download failed with status {response.status_code}")
            exit()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        exit()

def reorganize_subset(subset_name):
    subset_path = os.path.join(BASE_DIR, subset_name)
    csv_path = os.path.join(subset_path, '_classes.csv')

    # 1. Verification
    if not os.path.exists(csv_path):
        print(f"Warning: {subset_name} CSV not found at {csv_path}")
        return

    print(f"Organizing {subset_name}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Auto-detect filename column
    filename_col = 'filename'
    if filename_col not in df.columns:
        filename_col = df.columns[0]

    count_moved = 0

    # 2. Iterate through CSV and move files
    for index, row in df.iterrows():
        img_file = row[filename_col]
        src_path = os.path.join(subset_path, img_file)

        # Determine the class
        class_folder = "unknown"
        for csv_header, folder_name in label_map.items():
            # Check if column exists and is set to 1
            if csv_header in row and row[csv_header] == 1:
                class_folder = folder_name
                break

        # 3. Move the file
        if class_folder != "unknown" and os.path.exists(src_path):
            # Create the class folder (e.g., train/early_sign)
            dest_dir = os.path.join(subset_path, class_folder)
            os.makedirs(dest_dir, exist_ok=True)

            dest_path = os.path.join(dest_dir, img_file)
            shutil.move(src_path, dest_path)
            count_moved += 1

    print(f"Finished {subset_name}: Moved {count_moved} images.")

if __name__ == "__main__":
    # 1. Download First
    download_data_if_needed()

    # 2. Then Organize
    reorganize_subset('train')
    reorganize_subset('valid')
    reorganize_subset('test')

    print("\n--- SETUP COMPLETE ---")
    print(f"Data is ready for YOLO at: {BASE_DIR}")
    print("You can now run train_yolo.py")