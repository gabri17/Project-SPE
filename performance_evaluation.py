import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score, roc_curve  # Added missing imports
)
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
from scipy import stats
import joblib
from tqdm import tqdm
import warnings
from pandas.api.types import CategoricalDtype
from sklearn.exceptions import ConvergenceWarning


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def load_and_combine_datasets():
    base_path = os.path.join("datasets", "CSV Files")
    
    # Load datasets directly without feature names file
    train_file = os.path.join(base_path, 'Training and Testing Sets', 'UNSW_NB15_training-set.csv')
    test_file = os.path.join(base_path, 'Training and Testing Sets', 'UNSW_NB15_testing-set.csv')
    
    print("Loading training dataset...")
    train_df = pd.read_csv(
        train_file, 
        header=0,  # Use header row for column names
        low_memory=False,
        na_values=['', ' ', 'NA', 'N/A', 'NaN', 'nan', '-'],
        encoding='utf-8'
    )
    
    print("Loading testing dataset...")
    test_df = pd.read_csv(
        test_file, 
        header=0,  # Use header row for column names
        low_memory=False,
        na_values=['', ' ', 'NA', 'N/A', 'NaN', 'nan', '-'],
        encoding='utf-8'
    )
    
    # Standardize column names
    train_df.columns = [col.strip().lower() for col in train_df.columns]
    test_df.columns = [col.strip().lower() for col in test_df.columns]
    
    # Add dataset identifier
    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'
    
    # Combine both datasets
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    return combined_df


def preprocess_data(df):
    # Create attack indicator first
    if 'attack_cat' in df.columns:
        # Convert 'Normal' to NaN for attack detection
        df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
        df['is_attack'] = (df['attack_cat'] != 'Normal').astype(int)
    elif 'label' in df.columns:
        # Fallback to label if attack_cat is missing
        df['is_attack'] = df['label']
    else:
        raise KeyError("Neither 'attack_cat' nor 'label' column found in dataset")
    
    # Convert label column if present
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)
    
    print("\n" + "="*50)
    print("Initial Class Distribution:")
    print(f"Normal samples: {(df['is_attack'] == 0).sum()}")
    print(f"Attack samples: {(df['is_attack'] == 1).sum()}")
    print("="*50 + "\n")
    
    # Handle datetime features
    if 'stime' in df.columns and 'ltime' in df.columns:
        # Convert to numeric first
        df['stime'] = pd.to_numeric(df['stime'], errors='coerce')
        df['ltime'] = pd.to_numeric(df['ltime'], errors='coerce')
        
        # Now convert to datetime
        df['stime'] = pd.to_datetime(df['stime'], unit='s', errors='coerce')
        df['ltime'] = pd.to_datetime(df['ltime'], unit='s', errors='coerce')
        
        # Calculate duration and time features
        df['duration'] = (df['ltime'] - df['stime']).dt.total_seconds().fillna(0)
        df['hour'] = df['stime'].dt.hour.fillna(0)
        df['day_of_week'] = df['stime'].dt.dayofweek.fillna(0)
        
        df.drop(columns=['stime', 'ltime'], inplace=True)
    
    # Remove IP-related columns
    ip_related_cols = ['srcip', 'dstip', 'sport', 'dsport', 'id']
    for col in ip_related_cols:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Clean and convert categorical columns
    categorical_cols = ['proto', 'service', 'state', 'attack_cat', 'ct_flw_http_mthd']
    for col in categorical_cols:
        if col in df.columns:
            # Convert to string and clean
            df[col] = df[col].astype(str).str.strip()
            # Replace '-' and 'nan' with 'missing'
            df[col] = df[col].replace(['-', 'nan', 'NaN', 'N/A', 'NA', 'none'], 'missing')
            # Fill any remaining NaNs
            df[col] = df[col].fillna('missing')
    
    # Handle specific columns
    if 'ct_ftp_cmd' in df.columns:
        df['ct_ftp_cmd'] = pd.to_numeric(df['ct_ftp_cmd'], errors='coerce').fillna(0)
    
    if 'service' in df.columns:
        df['service'] = df['service'].fillna('None')
    
    if 'is_sm_ips_ports' in df.columns:
        df['is_sm_ips_ports'] = pd.to_numeric(df['is_sm_ips_ports'], errors='coerce').fillna(0).astype(np.int8)
    
    if 'is_ftp_login' in df.columns:
        df['is_ftp_login'] = pd.to_numeric(df['is_ftp_login'], errors='coerce').fillna(0).astype(np.int8)
    
    # Convert all columns to numeric types
    feature_cols = [col for col in df.columns if col not in ['attack_cat', 'is_attack', 'dataset']]
    
    for col in feature_cols:
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        # Convert object columns to numeric
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except:
            # Handle categorical conversion
            df[col] = df[col].astype('category').cat.codes.replace({-1: 0})
        
        # Fill any resulting NaNs
        if df[col].isna().any():
            df[col] = df[col].fillna(0)
    
    # Final check for non-numeric columns
    non_numeric = df[feature_cols].select_dtypes(include=['object']).columns
    if len(non_numeric) > 0:
        print(f"Converting remaining non-numeric columns: {list(non_numeric)}")
        for col in non_numeric:
            df[col] = df[col].astype('category').cat.codes
    
    # Downcast numeric types to save memory
    for col in df.select_dtypes(include=['int', 'int64']):
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include=['float', 'float64']):
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    print("\nClass distribution after preprocessing:")
    print(df['is_attack'].value_counts())
    
    return df


def scale_features(train_df, test_df):
    """Scale features using Min-Max scaling"""
    scaler = MinMaxScaler()
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col not in ['attack_cat', 'is_attack', 'dataset']]
    
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    return train_df, test_df, scaler


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
            print(f"Loaded pre-trained model and metrics for {model_name}")
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
    trained_models = {}  # Store trained models
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in tqdm(models.items(), desc="Training models"):
        model, metrics = load_or_train_model(name, model, X_train, y_train, X_test, y_test, task_type)
        trained_models[name] = model  # Save trained model
        
        os.makedirs("results", exist_ok=True)
        cv_file = f"results/{name.replace(' ', '_').lower()}_{task_type}_cv.pkl"
        if os.path.exists(cv_file):
            cv_scores = joblib.load(cv_file)
            print(f"Loaded CV scores for {name}")
        else:
            cv_scores = cross_validate_model(model, X_train, y_train, kf, task_type)
            joblib.dump(cv_scores, cv_file)
        
        metrics['cv_scores'] = cv_scores
        results[name] = metrics
    
    return trained_models, results  


def cross_validate_model(model, X, y, kf, task_type='binary'):
    from joblib import Parallel, delayed
    
    def train_fold(model, X_train, y_train, X_val, y_val):
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)
        y_prob = model_clone.predict_proba(X_val)[:, 1] if hasattr(model_clone, 'predict_proba') else None
        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='binary', zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_prob) if y_prob is not None else None
        }
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    fold_results = Parallel(n_jobs=-1)(
        delayed(train_fold)(model, X_arr[train_idx], y_arr[train_idx], X_arr[val_idx], y_arr[val_idx])
        for train_idx, val_idx in kf.split(X_arr, y_arr)
    )
    
    cv_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    for fold in fold_results:
        for metric in cv_results:
            if fold[metric] is not None:
                cv_results[metric].append(fold[metric])
    
    return cv_results


def calculate_metrics(y_true, y_pred, y_prob=None, task_type='binary'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    
    return metrics



def calculate_confidence_interval(scores, confidence=0.95):
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    return mean, mean - h, mean + h


def perform_statistical_analysis(results, metric='f1'):
    from scipy.stats import ttest_rel
    
    model_scores = {}
    for model_name, data in results.items():
        if 'cv_scores' in data and metric in data['cv_scores']:
            model_scores[model_name] = data['cv_scores'][metric]
    
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    p_values = np.zeros((n_models, n_models))
    effect_sizes = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(i+1, n_models):
            scores_i = model_scores[model_names[i]]
            scores_j = model_scores[model_names[j]]

            t_stat, p_val = ttest_rel(scores_i, scores_j)
            p_values[i, j] = p_val
            p_values[j, i] = p_val
            
            mean_diff = np.mean(scores_i) - np.mean(scores_j)
            pooled_std = np.sqrt((np.std(scores_i)**2 + np.std(scores_j)**2) / 2)
            effect_sizes[i, j] = mean_diff / pooled_std
            effect_sizes[j, i] = -effect_sizes[i, j]
    
    p_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
    es_df = pd.DataFrame(effect_sizes, index=model_names, columns=model_names)
    
    p_df.to_csv(f"results/{metric}_pairwise_pvalues.csv")
    es_df.to_csv(f"results/{metric}_pairwise_effect_sizes.csv")
    
    print(f"\nSignificant differences in {metric} (p < 0.05):")
    for i in range(n_models):
        for j in range(i+1, n_models):
            if p_values[i, j] < 0.05:
                print(f"{model_names[i]} vs {model_names[j]}: p={p_values[i, j]:.4f}, d={effect_sizes[i, j]:.2f}")
    
    return p_df, es_df


def plot_statistical_results(results, metric='f1'):
    model_names = []
    metric_means = []
    ci_lowers = []
    ci_uppers = []
    
    for model_name, data in results.items():
        if 'cv_scores' in data and metric in data['cv_scores']:
            scores = data['cv_scores'][metric]
            mean, ci_lower, ci_upper = calculate_confidence_interval(scores)
            model_names.append(model_name)
            metric_means.append(mean)
            ci_lowers.append(ci_lower)
            ci_uppers.append(ci_upper)
    
    sorted_idx = np.argsort(metric_means)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    metric_means = [metric_means[i] for i in sorted_idx]
    ci_lowers = [ci_lowers[i] for i in sorted_idx]
    ci_uppers = [ci_uppers[i] for i in sorted_idx]
    
    plt.figure(figsize=(12, 8))
    y_pos = np.arange(len(model_names))
    
    plt.barh(y_pos, metric_means, xerr=[np.array(metric_means) - np.array(ci_lowers), 
                                        np.array(ci_uppers) - np.array(metric_means)], 
             align='center', alpha=0.7, color='skyblue')
    
    plt.yticks(y_pos, model_names)
    plt.xlabel(f'{metric.capitalize()} Score')
    plt.title(f'Model Performance Comparison with 95% CIs ({metric.capitalize()})')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(metric_means):
        plt.text(v + 0.01, i, f"{v:.3f}", color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(f"results/{metric}_comparison_with_ci.png")
    plt.close()


def plot_confusion_matrices(results, y_true):
    plt.figure(figsize=(15, 10))
    n_models = len(results)
    
    for i, (model_name, data) in enumerate(results.items(), 1):
        cm = data['confusion_matrix']
        plt.subplot(2, 3, i)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Attack'],
                    yticklabels=['Normal', 'Attack'])
        
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices_binary.png")
    plt.close()


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    print("Loading and preprocessing data...")
    df = load_and_combine_datasets()
    processed_df = preprocess_data(df)
    
    # Split back into train and test
    train_df = processed_df[processed_df['dataset'] == 'train']
    test_df = processed_df[processed_df['dataset'] == 'test']
    
    train_df = train_df.drop(columns=['dataset'])
    test_df = test_df.drop(columns=['dataset'])
    
    # Prepare features and target
    X_train = train_df.drop(columns=['attack_cat', 'is_attack', 'label'])
    y_train = train_df['is_attack']
    
    X_test = test_df.drop(columns=['attack_cat', 'is_attack', 'label'])
    y_test = test_df['is_attack']
    
    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    print("\nRunning intrusion detection (binary classification)...")
    # Capture both return values
    trained_models, binary_results = train_and_evaluate(MODELS, X_train, y_train, X_test, y_test, 'binary')
    
    print("\nPerforming statistical analysis...")
    perform_statistical_analysis(binary_results, 'f1')
    plot_statistical_results(binary_results, 'f1')
    plot_confusion_matrices(binary_results, y_test)
    
    results_data = []
    for model_name, metrics in binary_results.items():
        row = {
            'Model': model_name,
            'Train Time (s)': metrics.get('train_time', 'N/A'),
            'Test Accuracy': metrics['accuracy'],
            'Test Precision': metrics['precision'],
            'Test Recall': metrics['recall'],
            'Test F1': metrics['f1'],
            'Test AUC': metrics.get('roc_auc', 'N/A'),
            'CV F1 Mean': np.mean(metrics['cv_scores']['f1']),
            'CV F1 Std': np.std(metrics['cv_scores']['f1'])
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("results/intrusion_detection_results.csv", index=False)
    

    print("\nGenerating classification reports...")
    for model_name in MODELS.keys():
        try:
            model = joblib.load(f"models/{model_name.replace(' ', '_').lower()}_binary.pkl")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f"results/{model_name}_classification_report.csv")
            
            if hasattr(model, 'predict_proba'):
                from sklearn.metrics import RocCurveDisplay
                RocCurveDisplay.from_predictions(y_test, model.predict_proba(X_test)[:, 1])
                plt.title(f'ROC Curve - {model_name}')
                plt.savefig(f"results/{model_name}_roc_curve.png")
                plt.close()
        except Exception as e:
            print(f"Could not generate report for {model_name}: {str(e)}")
    
    print("\n=== Intrusion Detection Complete ===")
    print(f"Processed {len(processed_df)} records ({y_test.sum()} attacks, {len(y_test)-y_test.sum()} normal)")
    print("Results saved to results/ directory")

if __name__ == "__main__":
    main()