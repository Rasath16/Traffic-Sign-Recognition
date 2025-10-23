"""
Debug script to check GTSRB dataset structure
"""
import pandas as pd
import os

# Paths
TRAIN_CSV = 'data/raw/Train.csv'
TEST_CSV = 'data/raw/Test.csv'
TRAIN_DIR = 'data/raw/Train'
TEST_DIR = 'data/raw/Test'

print("=" * 70)
print("DEBUGGING GTSRB DATASET")
print("=" * 70)

# Check if files exist
print("\n[1] Checking if files exist...")
print(f"Train.csv exists: {os.path.exists(TRAIN_CSV)}")
print(f"Test.csv exists: {os.path.exists(TEST_CSV)}")
print(f"Train/ folder exists: {os.path.exists(TRAIN_DIR)}")
print(f"Test/ folder exists: {os.path.exists(TEST_DIR)}")

# Check CSV structure
if os.path.exists(TRAIN_CSV):
    print("\n[2] Train.csv structure:")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"Number of rows: {len(df_train)}")
    print(f"Columns: {df_train.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df_train.head())
    print("\nSample paths:")
    if 'Path' in df_train.columns:
        print(df_train['Path'].head())
    
    # Check if images exist
    print("\n[3] Checking if sample images exist...")
    for i in range(min(5, len(df_train))):
        if 'Path' in df_train.columns:
            img_path = os.path.join(TRAIN_DIR, df_train.iloc[i]['Path'])
            exists = os.path.exists(img_path)
            print(f"  {df_train.iloc[i]['Path']}: {exists}")

if os.path.exists(TEST_CSV):
    print("\n[4] Test.csv structure:")
    df_test = pd.read_csv(TEST_CSV)
    print(f"Number of rows: {len(df_test)}")
    print(f"Columns: {df_test.columns.tolist()}")
    print("\nFirst 5 rows:")
    print(df_test.head())

# Check folder structure
print("\n[5] Checking Train folder structure...")
if os.path.exists(TRAIN_DIR):
    train_contents = os.listdir(TRAIN_DIR)
    print(f"Items in Train/: {len(train_contents)}")
    print(f"First 10 items: {train_contents[:10]}")
    
    # Check if it has class subfolders
    class_folders = [f for f in train_contents if os.path.isdir(os.path.join(TRAIN_DIR, f)) and f.isdigit()]
    print(f"Class folders (0-42): {len(class_folders)}")
    if class_folders:
        print(f"Sample class folders: {sorted(class_folders)[:10]}")
        
        # Check contents of a class folder
        sample_class = class_folders[0]
        sample_path = os.path.join(TRAIN_DIR, sample_class)
        sample_images = os.listdir(sample_path)
        print(f"\nImages in class {sample_class}/: {len(sample_images)}")
        print(f"Sample images: {sample_images[:5]}")

print("\n" + "=" * 70)
print("Debug complete!")
print("=" * 70)