import os
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_and_combine_datasets():
    base_path = os.path.join("datasets", "CSV Files")
    training_set_file = os.path.join(base_path, 'Training and Testing Sets', 'UNSW_NB15_training-set.csv')
    testing_set_file = os.path.join(base_path, 'Training and Testing Sets', 'UNSW_NB15_testing-set.csv')

    train_df = pd.read_csv(training_set_file, header=0, low_memory=False, na_values=['', ' ', 'NA', 'N/A', 'NaN', 'nan', '-'], encoding='utf-8')
    test_df = pd.read_csv(testing_set_file, header=0, low_memory=False, na_values=['', ' ', 'NA', 'N/A', 'NaN', 'nan', '-'], encoding='utf-8')
    
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    
    return pd.concat([train_df, test_df], ignore_index=True)


def preprocess_data(df):
    if 'attack_cat' in df.columns:
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        df['is_attack'] = (df['attack_cat'] != 'Normal').astype(int)
    elif 'label' in df.columns:
        df['is_attack'] = df['label']

    
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    print("Initial Class Distribution:")
    print(f"Normal: {(df['is_attack'] == 0).sum()}")
    print(f"Attack: {(df['is_attack'] == 1).sum()}")
    
    if 'stime' in df.columns and 'ltime' in df.columns:
        df['stime'] = pd.to_numeric(df['stime'], errors='coerce')
        df['ltime'] = pd.to_numeric(df['ltime'], errors='coerce')
        df['stime'] = pd.to_datetime(df['stime'], unit='s', errors='coerce')
        df['ltime'] = pd.to_datetime(df['ltime'], unit='s', errors='coerce')
        df['duration'] = (df['ltime'] - df['stime']).dt.total_seconds().fillna(0)
        df['hour'] = df['stime'].dt.hour.fillna(0)
        df['day_of_week'] = df['stime'].dt.dayofweek.fillna(0)
        df.drop(columns=['stime', 'ltime'], inplace=True)
    
    ip_columns = ['srcip', 'dstip', 'sport', 'dsport', 'id']
    for col in ip_columns:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    categorical_cols = ['proto', 'service', 'state', 'attack_cat', 'ct_flw_http_mthd']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['-', 'nan', 'NaN', 'N/A', 'NA', 'none'], 'missing')
            df[col] = df[col].fillna('missing')
    

    col_names = ['is_sm_ips_ports','is_ftp_login','ct_ftp_cmd']
    for col in col_names:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(np.int8)
    if 'service' in df.columns:
        df['service'] = df['service'].fillna('None')
  
    
    feature_cols = [col for col in df.columns if col not in ['attack_cat', 'is_attack', 'dataset']]
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            df[col] = df[col].astype('category').cat.codes.replace({-1: 0})
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    non_numeric = df[feature_cols].select_dtypes(include=['object']).columns
    for col in non_numeric:
        df[col] = df[col].astype('category').cat.codes
    
    for col in df.select_dtypes(include=['int', 'int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float', 'float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    print("\nClass distribution after preprocessing:")
    print(df['is_attack'].value_counts())
    return df

def scale_features(train_df, test_df):
    scaler = MinMaxScaler()
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col not in ['attack_cat', 'is_attack', 'dataset']]
    
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    return train_df, test_df, scaler