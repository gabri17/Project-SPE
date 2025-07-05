import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from config import MODELS, CV_SPLITS, RANDOM_STATE, ANALYSIS_DIR
import matplotlib.pyplot as plt

class ModelTrainer:
    """Handles model training and evaluation"""
    def __init__(self, X_train, X_test, y_train, y_test, feature_names, problem_type='binary'):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.problem_type = problem_type
        self.models = {}
        self.results = {}
        self.cv_results = {}
        
    def train_models(self):
        """Train all models with cross-validation"""
        cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        
        if self.problem_type == 'multiclass':
            scoring = {m: f'{m}_macro' for m in ['accuracy', 'precision', 'recall', 'f1']}
        
        for name, params in MODELS[self.problem_type].items():
            print(f"\nTraining {name} for {self.problem_type} classification...")
            
            # Initialize model
            if name == "Logistic Regression":
                model = LogisticRegression(**params, random_state=RANDOM_STATE)
            elif name == "Random Forest":
                model = RandomForestClassifier(**params, random_state=RANDOM_STATE)
            elif name == "Neural Network":
                model = MLPClassifier(**params, random_state=RANDOM_STATE)
            elif name == "Naive Bayes":
                model = GaussianNB(**params)
            elif name == "Decision Tree":
                model = DecisionTreeClassifier(**params, random_state=RANDOM_STATE)
            else:
                continue
                
            # Cross-validation
            start_time = time.time()
            cv_res = cross_validate(
                model, self.X_train, self.y_train, 
                cv=cv, scoring=scoring, n_jobs=-1,
                return_train_score=False
            )
            cv_time = time.time() - start_time
            
            # Store CV results
            self.cv_results[name] = cv_res
            cv_metrics = {}
            for metric in scoring:
                if isinstance(scoring, dict):
                    metric_key = f'test_{metric}'
                else:
                    metric_key = f'test_{metric}'
                values = cv_res[metric_key]
                cv_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'values': values
                }
            
            # Train final model
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - start_time
            
            # Test evaluation
            start_time = time.time()
            y_pred = model.predict(self.X_test)
            inference_time = (time.time() - start_time) / len(self.X_test)
            
            # Calculate metrics
            if self.problem_type == 'binary':
                test_metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred),
                    'recall': recall_score(self.y_test, y_pred),
                    'f1': f1_score(self.y_test, y_pred)
                }
            else:
                test_metrics = {
                    'accuracy': accuracy_score(self.y_test, y_pred),
                    'precision': precision_score(self.y_test, y_pred, average='macro'),
                    'recall': recall_score(self.y_test, y_pred, average='macro'),
                    'f1': f1_score(self.y_test, y_pred, average='macro')
                }
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            self.plot_confusion_matrix(cm, name)
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                self.plot_feature_importance(model, name)
            
            # Store results
            self.models[name] = model
            self.results[name] = {
                'cv_metrics': cv_metrics,
                'test_metrics': test_metrics,
                'train_time': train_time,
                'inference_time': inference_time,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"Completed {name} in {cv_time:.2f}s (CV) + {train_time:.2f}s (training)")
            
        return self.results
    
    def plot_confusion_matrix(self, cm, model_name):
        """Generate and save confusion matrix"""
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix: {model_name} ({self.problem_type})')
        plt.savefig(f"{ANALYSIS_DIR}/confusion_matrix_{model_name}_{self.problem_type}.png")
        plt.close()
    
    def plot_feature_importance(self, model, model_name):
        """Generate and save feature importance plot"""
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            return
            
        indices = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(12, 8))
        plt.title(f"Feature Importance: {model_name} ({self.problem_type})")
        plt.bar(range(len(indices)), importances[indices], align='center')
        plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/feature_importance_{model_name}_{self.problem_type}.png")
        plt.close()