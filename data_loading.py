import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from config import BASE_PATH, PART_FILES, TEST_SIZE, RANDOM_STATE

# Official UNSW-NB15 column names
UNSW_COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
    'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 
    'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
    'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 
    'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
    'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
    'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
    'ct_dst_src_ltm', 'attack_cat', 'label'
]

def load_and_combine_datasets():
    """Load and combine the four dataset files with correct column names"""
    dfs = []
    for file in PART_FILES:
        file_path = Path(BASE_PATH) / file
        print(f"Loading {file_path}...")
        
        if not file_path.exists():
            available = [f.name for f in Path(BASE_PATH).iterdir()]
            print(f"File not found! Available files: {available}")
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load with correct column names
        try:
            df_part = pd.read_csv(file_path, header=None, names=UNSW_COLUMNS, low_memory=False)
            print(f"  Success: Loaded {len(df_part)} rows with {len(df_part.columns)} columns")
            dfs.append(df_part)
        except Exception as e:
            print(f"  Error loading {file_path}: {str(e)}")
            raise
    
    # Combine all parts
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(df_full)} rows, {df_full.shape[1]} columns")
    print(f"Columns: {df_full.columns.tolist()}")
    
    # Verify we have required columns
    if 'label' not in df_full.columns:
        print("\nCRITICAL: 'label' column not found after processing")
        print("Columns available:", df_full.columns.tolist())
        raise ValueError("Binary target column 'label' not found in dataset")
    
    return df_full

def clean_dataset(df):
    """Perform data cleaning operations"""
    print("\nCleaning dataset...")
    
    # Handle missing values
    initial_rows = len(df)
    df.replace([np.inf, -np.inf, 'inf', 'Infinity'], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    final_rows = len(df)
    
    print(f"  Removed {initial_rows - final_rows} rows (missing values/duplicates)")
    print(f"  Final clean dataset: {final_rows} rows")
    
    return df

def create_train_test_split(df_full):
    """Create stratified train-test split"""
    print("\nCreating train/test split...")
    
    # Create multiclass target
    df_full['attack_cat_num'] = pd.factorize(df_full['attack_cat'])[0]
    stratify_cols = ['label', 'attack_cat_num']
    
    # Stratified split
    df_train, df_test = train_test_split(
        df_full, 
        test_size=TEST_SIZE, 
        stratify=df_full[stratify_cols],
        random_state=RANDOM_STATE
    )
    
    print(f"  Training set: {len(df_train)} samples")
    print(f"  Testing set: {len(df_test)} samples")
    
    return df_train, df_test

def load_datasets():
    """Main function to handle dataset loading"""
    # Load and combine datasets
    df_full = load_and_combine_datasets()
    
    # Clean dataset
    df_full = clean_dataset(df_full)
    
    # Create train/test split
    df_train, df_test = create_train_test_split(df_full)
    
    print(f"\nFinal training set: {df_train.shape}")
    print(f"Final testing set: {df_test.shape}")
    
    return df_train, df_test