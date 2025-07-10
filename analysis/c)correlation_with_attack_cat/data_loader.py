import pandas as pd
import os

class UNSWNB15Loader:

    def __init__(self, base_path=None, files=None, features_file=None):
        if base_path is None:
            base_path = os.path.join("datasets", "CSV_Files")
        if files is None:
            files = [
                'UNSW-NB15_1.csv',
                'UNSW-NB15_2.csv',
                'UNSW-NB15_3.csv',
                'UNSW-NB15_4.csv'
            ]
        if features_file is None:
            features_file = 'NUSW-NB15_features.csv'
        self.base_path = base_path
        self.files = files
        self.features_file = features_file
        self.names = self._load_feature_names()

    def _load_feature_names(self):
        return pd.read_csv(
            os.path.join(self.base_path, self.features_file), encoding='cp1252'
        )['Name'].str.strip()

    def load(self):
        dataframes = []
        
        for _, file in enumerate(self.files):
            file_path = os.path.join(self.base_path, file)
            df = pd.read_csv(file_path, header=None, low_memory=False)
            dataframes.append(df)
        
        df_all = pd.concat(dataframes, ignore_index=True)
        df_all.columns = self.names
        df_all = df_all.rename(columns={'Label': 'label'})
        return df_all

# Usage example:
""" base_path = os.path.join("datasets", "CSV_Files")
files = [
    'UNSW-NB15_1.csv',
    'UNSW-NB15_2.csv',
    'UNSW-NB15_3.csv',
    'UNSW-NB15_4.csv'
]
features_file = 'NUSW-NB15_features.csv'
 """
##### Example code
""" loader = UNSWNB15Loader()
df = loader.load()
print(f"Loaded dataset with {len(df)} rows and {df.shape[1]} columns.") """