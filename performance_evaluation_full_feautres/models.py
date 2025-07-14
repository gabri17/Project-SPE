import os
import time
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

MODELS = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, verbose=0),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, verbose=0)
}

def load_or_train_model(model_name, model, X_train, y_train, X_test, y_test, task_type):
    os.makedirs("models", exist_ok=True)
    model_file = f"models/{model_name.replace(' ', '_').lower()}_{task_type}.pkl"
    results_file = f"results/{model_name.replace(' ', '_').lower()}_{task_type}_metrics.pkl"
    
    if os.path.exists(model_file) and os.path.exists(results_file):
        try:
            model = joblib.load(model_file)
            metrics = joblib.load(results_file)
            print(f"Loaded pre-trained model for {model_name}")
            return model, metrics
        except:
            print(f"Error loading {model_name}, retraining...")
    
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = calculate_metrics(y_test, y_pred, y_prob, task_type)
    metrics['train_time'] = train_time
    
    joblib.dump(model, model_file)
    joblib.dump(metrics, results_file)
    
    return model, metrics

def train_and_evaluate(models, X_train, y_train, X_test, y_test, task_type='binary'):
    results = {}
    trained_models = {}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in tqdm(models.items(), desc="Training models"):
        model, metrics = load_or_train_model(name, model, X_train, y_train, X_test, y_test, task_type)
        trained_models[name] = model
        
        os.makedirs("results", exist_ok=True)
        cv_file = f"results/{name.replace(' ', '_').lower()}_{task_type}_cv.pkl"
        if os.path.exists(cv_file):
            cv_scores = joblib.load(cv_file)
            print(f"Loaded CV scores for {name}")
        else:
            cv_scores = cross_validation(model, X_train, y_train, kf)
            joblib.dump(cv_scores, cv_file)
        
        metrics['cv_scores'] = cv_scores
        results[name] = metrics
    
    return trained_models, results

def cross_validation(model, X, y, kf):
    X_arr = getattr(X, "values", X)
    y_arr = getattr(y, "values", y)

    metrics = ['accuracy','precision','recall','f1','roc_auc']
    all_scores = {m: [] for m in metrics}

    fold_scores = Parallel(n_jobs=-1)(
        delayed(do_fold)(model, X_arr[tr], y_arr[tr], X_arr[va], y_arr[va])
        for tr, va in kf.split(X_arr, y_arr)
    )

    for scores in fold_scores:
        for m in metrics:
            if scores[m] is not None:
                all_scores[m].append(scores[m])

    return all_scores

def do_fold(model, X_tr, y_tr, X_va, y_va):
    m = clone(model)
    m.fit(X_tr, y_tr)

    y_pred = m.predict(X_va)
    y_prob = m.predict_proba(X_va)[:,1] if hasattr(m, 'predict_proba') else None

    return {
        'accuracy':  accuracy_score(y_va, y_pred),
        'precision': precision_score(y_va, y_pred, zero_division=0),
        'recall':    recall_score(y_va, y_pred, zero_division=0),
        'f1':        f1_score(y_va, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_va, y_prob) if y_prob is not None else None
    }

def calculate_metrics(y_true, y_pred, y_prob=None, task_type='binary'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=task_type, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=task_type, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=task_type, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)

    return metrics