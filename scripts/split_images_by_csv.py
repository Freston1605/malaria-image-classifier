import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIGURATION
DATA_DIR = 'data'
CSV_PATH = os.path.join(DATA_DIR, 'train_data.csv')
IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')
OUTPUT_DIR = 'dataset'

# Map numeric labels to descriptive folder names
label_map = {
    0: 'healthy',
    1: 'infected'
}

# Read CSV (assumes columns: id,label)
df = pd.read_csv(CSV_PATH)
df['label'] = df['label'].astype(int)
df['label_name'] = df['label'].map(label_map)

# Split into train/val
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df['label'], random_state=42
)

splits = {
    'train': train_df,
    'val': val_df
}

# Create output directories and copy images
for split, split_df in splits.items():
    for label_name in split_df['label_name'].unique():
        os.makedirs(os.path.join(OUTPUT_DIR, split, label_name), exist_ok=True)
    for _, row in split_df.iterrows():
        src = os.path.join(IMAGES_DIR, row['img_name'])
        dst = os.path.join(OUTPUT_DIR, split, row['label_name'], row['img_name'])
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            print(f"Warning: {src} does not exist.")

print("Image splitting and organization complete.")
