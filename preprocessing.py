from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from config import CORRELATION_THRESHOLD

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Custom transformer for feature selection"""
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features]

class DataPreprocessor:
    """Handles all data preprocessing tasks"""
    def __init__(self, target_binary='label', target_multiclass='attack_cat'):
        self.target_binary = target_binary
        self.target_multiclass = target_multiclass
        self.numeric_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.feature_names = None
        
    def prepare_data(self, df_train, df_test, has_multiclass=True):
        """Prepare datasets for training"""
        # Identify feature types
        self.numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove targets from features
        for col in [self.target_binary, self.target_multiclass, 'attack_cat']:
            if col in self.numeric_features:
                self.numeric_features.remove(col)
            if col in self.categorical_features:
                self.categorical_features.remove(col)
        
        # Create transformers
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer([
            ('num', numeric_transformer, self.numeric_features),
            ('cat', categorical_transformer, self.categorical_features)
        ])
        
        # Prepare datasets
        X_train = df_train.drop(columns=[self.target_binary])
        y_train_bin = df_train[self.target_binary]
        
        X_test = df_test.drop(columns=[self.target_binary])
        y_test_bin = df_test[self.target_binary]
        
        # Handle multiclass if available
        if has_multiclass and self.target_multiclass in df_train.columns:
            y_train_multi = df_train[self.target_multiclass]
            y_test_multi = df_test[self.target_multiclass]
        else:
            y_train_multi = None
            y_test_multi = None
        
        # Fit preprocessor
        self.preprocessor.fit(X_train)
        self.feature_names = self.get_feature_names()
        
        return (
            self.preprocessor.transform(X_train),
            self.preprocessor.transform(X_test),
            y_train_bin, y_test_bin,
            y_train_multi, y_test_multi
        )
    
    def get_feature_names(self):
        """Get feature names after preprocessing"""
        num_names = self.numeric_features
        cat_names = self.preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(self.categorical_features)
        return np.concatenate([num_names, cat_names])