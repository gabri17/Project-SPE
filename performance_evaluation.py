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
    roc_auc_score, confusion_matrix, classification_report
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

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ================================================
# Optimized Data Loading and Preprocessing
# ================================================

def load_and_combine_datasets():
    """Optimized dataset loading with data types specification"""
    base_path = os.path.join("datasets", "CSV Files")
    file_names = [ 
        'UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 
        'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv'
    ]
    
    # Load feature names and convert to lowercase
    features_file = os.path.join(base_path, 'NUSW-NB15_features.csv')
    features_df = pd.read_csv(features_file, encoding='cp1252')
    column_names = [name.strip().lower() for name in features_df['Name']]
    
    # Optimized data types for memory efficiency
    dtype_spec = {
        'srcip': 'category',
        'dstip': 'category',
        'proto': 'category',
        'service': 'category',
        'state': 'category',
        'attack_cat': 'category'
    }
    
    # Load and combine datasets with progress tracking
    dfs = []
    for file in tqdm(file_names, desc="Loading datasets"):
        file_path = os.path.join(base_path, file)
        df_part = pd.read_csv(
            file_path, 
            header=None, 
            names=column_names, 
            low_memory=False,
            dtype={col: dtype_spec.get(col, 'float32') for col in column_names if col in dtype_spec}
        )
        dfs.append(df_part)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Downcast numeric columns to save memory
    for col in combined_df.select_dtypes(include=['int', 'int64']):
        combined_df[col] = pd.to_numeric(combined_df[col], downcast='integer')
    for col in combined_df.select_dtypes(include=['float', 'float64']):
        combined_df[col] = pd.to_numeric(combined_df[col], downcast='float')
    
    return combined_df

def preprocess_data(df):
    """Preprocess the combined dataset for attack-type classification"""
    # Convert timestamp features to datetime
    df['stime'] = pd.to_datetime(df['stime'], unit='s', errors='coerce')
    df['ltime'] = pd.to_datetime(df['ltime'], unit='s', errors='coerce')
    
    # Calculate duration from timestamps
    df['duration'] = (df['ltime'] - df['stime']).dt.total_seconds().fillna(0)
    
    # Create time-based features
    df['hour'] = df['stime'].dt.hour
    df['day_of_week'] = df['stime'].dt.dayofweek
    
    # Drop original timestamp columns
    df.drop(columns=['stime', 'ltime'], inplace=True)
    
    # Drop high cardinality and identifier columns
    cols_to_drop = ['srcip', 'dstip', 'sport', 'dsport', 'id']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    
    # Convert spaces to NaN and handle missing values
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].fillna(0)
    
    # Handle service column separately
    if 'service' in df.columns:
        # Convert to string type first if it's categorical
        if isinstance(df['service'].dtype, CategoricalDtype):
            df['service'] = df['service'].astype(str)
        df['service'] = df['service'].fillna('None')
    
    # Handle binary features
    df['is_sm_ips_ports'] = df['is_sm_ips_ports'].fillna(0).astype(np.int8)
    df['is_ftp_login'] = df['is_ftp_login'].fillna(0).astype(np.int8)
    
    # Optimized categorical encoding
    cat_cols = ['proto', 'service', 'state']
    for col in cat_cols:
        if col in df.columns:
            # Ensure we're working with strings
            df[col] = df[col].astype(str)
            df[col] = df[col].astype('category').cat.codes.replace({-1: 0})
    
    # ======================================================
    # ATTACK-TYPE CLASSIFICATION FOCUS
    # ======================================================
    
    # Clean and standardize attack names
    df['attack_cat'] = df['attack_cat'].astype(str).str.strip()
    
    # Replace the string 'nan' with actual NaN
    df['attack_cat'] = df['attack_cat'].replace('nan', np.nan)
    
    # Filter out normal traffic and missing categories
    attack_df = df[df['attack_cat'] != 'Normal']
    if attack_df['attack_cat'].isnull().any():
        print(f"Dropping {attack_df['attack_cat'].isnull().sum()} rows with missing attack_cat")
        attack_df = attack_df.dropna(subset=['attack_cat'])
    
    # Consolidate similar attack names
    attack_mapping = {
        'Fuzzers': 'Fuzzers',
        'Fuzzer': 'Fuzzers',
        'Fuzzers ': 'Fuzzers',
        ' Fuzzers': 'Fuzzers',
        'Reconnaissance': 'Reconnaissance',
        'Reconaissance': 'Reconnaissance',
        ' Reconnaissance': 'Reconnaissance',
        'Reconnaissance ': 'Reconnaissance',
        'Backdoor': 'Backdoor',
        'Backdoors': 'Backdoor',
        'Backdoor ': 'Backdoor',
        'Backdoors ': 'Backdoor',
        'Shellcode': 'Shellcode',
        'Shellcode ': 'Shellcode',
        ' Shellcode': 'Shellcode',
        'DoS': 'DoS',
        'DoS ': 'DoS',
        ' DoS': 'DoS',
        'Generic': 'Generic',
        'Generic ': 'Generic',
        'Exploits': 'Exploits',
        'Exploit': 'Exploits',
        'Exploits ': 'Exploits',
        'Analysis': 'Analysis',
        'Analytics': 'Analysis',
        'Worms': 'Worms',
        'Worm': 'Worms'
    }
    attack_df['attack_type'] = attack_df['attack_cat'].map(attack_mapping).fillna(attack_df['attack_cat'])
    
    # Filter to only attacks with sufficient samples
    min_samples = 1000  # Minimum samples per attack type
    attack_counts = attack_df['attack_type'].value_counts()
    valid_attacks = attack_counts[attack_counts >= min_samples].index.tolist()
    attack_df = attack_df[attack_df['attack_type'].isin(valid_attacks)]
    
    # Print attack type distribution
    print("\nAttack types for classification:")
    print(attack_df['attack_type'].value_counts())
    print(f"\nTotal attack samples: {len(attack_df)}")
    
    # Fill residual NaNs in features
    feature_cols = [col for col in attack_df.columns if col not in ['attack_cat', 'attack_type']]
    attack_df[feature_cols] = attack_df[feature_cols].fillna(0)
    
    return attack_df

def scale_features(train_df, test_df):
    """Scale features using Min-Max scaling"""
    scaler = MinMaxScaler()
    num_cols = train_df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [col for col in num_cols if col not in ['attack_cat', 'attack_type']]
    
    train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df[num_cols] = scaler.transform(test_df[num_cols])
    
    return train_df, test_df, scaler

# ================================================
# Optimized Model Training and Evaluation
# ================================================

MODELS = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

def load_or_train_model(model_name, model, X_train, y_train, X_test, y_test, task_type):
    """Load model if exists, otherwise train and save it"""
    model_file = f"models/{model_name.replace(' ', '_').lower()}_{task_type}.pkl"
    results_file = f"results/{model_name.replace(' ', '_').lower()}_{task_type}_metrics.pkl"
    
    # Try to load existing model and metrics
    if os.path.exists(model_file) and os.path.exists(results_file):
        try:
            model = joblib.load(model_file)
            metrics = joblib.load(results_file)
            print(f"Loaded pre-trained model and metrics for {model_name}")
            return model, metrics
        except:
            print(f"Error loading {model_name}, retraining...")
    
    # Train new model if not found
    print(f"Training {model_name}...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Evaluate model
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    metrics = calculate_metrics(y_test, y_pred, y_prob, task_type)
    metrics['train_time'] = train_time
    
    # Save model and metrics
    joblib.dump(model, model_file)
    joblib.dump(metrics, results_file)
    
    return model, metrics

def train_and_evaluate(models, X_train, y_train, X_test, y_test, task_type='binary'):
    """Train and evaluate models with caching and advanced metrics"""
    results = {}
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in tqdm(models.items(), desc="Training models"):
        # Load or train model
        model, metrics = load_or_train_model(name, model, X_train, y_train, X_test, y_test, task_type)
        
        # Perform cross-validation only if not cached
        cv_file = f"results/{name.replace(' ', '_').lower()}_{task_type}_cv.pkl"
        if os.path.exists(cv_file):
            cv_scores = joblib.load(cv_file)
            print(f"Loaded CV scores for {name}")
        else:
            cv_scores = cross_validate_model(model, X_train, y_train, kf, task_type)
            joblib.dump(cv_scores, cv_file)
        
        metrics['cv_scores'] = cv_scores
        results[name] = metrics
    
    return results

def cross_validate_model(model, X, y, kf, task_type='binary'):
    """Optimized cross-validation with parallel processing"""
    from joblib import Parallel, delayed
    
    def train_fold(model, X_train, y_train, X_val, y_val):
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)
        return {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted' if task_type == 'multiclass' else 'binary', zero_division=0),
            'recall': recall_score(y_val, y_pred, average='weighted' if task_type == 'multiclass' else 'binary', zero_division=0),
            'f1': f1_score(y_val, y_pred, average='weighted' if task_type == 'multiclass' else 'binary', zero_division=0)
        }
    
    # Convert to NumPy arrays for consistent indexing
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    y_arr = y.values if isinstance(y, pd.Series) else y
    
    # Parallel execution of folds
    fold_results = Parallel(n_jobs=-1)(
        delayed(train_fold)(model, X_arr[train_idx], y_arr[train_idx], X_arr[val_idx], y_arr[val_idx])
        for train_idx, val_idx in kf.split(X_arr, y_arr)
    )
    
    # Aggregate results
    cv_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    for fold in fold_results:
        for metric in cv_results:
            cv_results[metric].append(fold[metric])
    
    return cv_results

def calculate_metrics(y_true, y_pred, y_prob=None, task_type='binary'):
    """Calculate evaluation metrics with proper averaging"""
    avg_method = 'binary' if task_type == 'binary' else 'weighted'
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=avg_method, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=avg_method, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=avg_method, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    if y_prob is not None:
        if task_type == 'binary':
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            try:
                # Use One-vs-Rest approach for multiclass ROC-AUC
                metrics['roc_auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='weighted'
                )
            except:
                metrics['roc_auc'] = None
    
    return metrics

# ================================================
# Advanced Statistical Analysis
# ================================================

def calculate_confidence_interval(scores, confidence=0.95):
    """Calculate confidence interval for metric scores"""
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    return mean, mean - h, mean + h

def perform_statistical_analysis(results, metric='f1'):
    """Perform t-tests and calculate effect sizes between models"""
    from scipy.stats import ttest_rel
    import numpy as np
    
    # Extract metric scores from cross-validation
    model_scores = {}
    for model_name, data in results.items():
        if 'cv_scores' in data and metric in data['cv_scores']:
            model_scores[model_name] = data['cv_scores'][metric]
    
    # Prepare results matrix
    model_names = list(model_scores.keys())
    n_models = len(model_names)
    p_values = np.zeros((n_models, n_models))
    effect_sizes = np.zeros((n_models, n_models))
    
    # Calculate pairwise comparisons
    for i in range(n_models):
        for j in range(i+1, n_models):
            scores_i = model_scores[model_names[i]]
            scores_j = model_scores[model_names[j]]
            
            # Paired t-test
            t_stat, p_val = ttest_rel(scores_i, scores_j)
            p_values[i, j] = p_val
            p_values[j, i] = p_val
            
            # Cohen's d effect size
            mean_diff = np.mean(scores_i) - np.mean(scores_j)
            pooled_std = np.sqrt((np.std(scores_i)**2 + np.std(scores_j)**2) / 2)
            effect_sizes[i, j] = mean_diff / pooled_std
            effect_sizes[j, i] = -effect_sizes[i, j]
    
    # Create DataFrames for results
    p_df = pd.DataFrame(p_values, index=model_names, columns=model_names)
    es_df = pd.DataFrame(effect_sizes, index=model_names, columns=model_names)
    
    # Save results
    p_df.to_csv(f"results/{metric}_pairwise_pvalues.csv")
    es_df.to_csv(f"results/{metric}_pairwise_effect_sizes.csv")
    
    # Print significant comparisons
    print(f"\nSignificant differences in {metric} (p < 0.05):")
    for i in range(n_models):
        for j in range(i+1, n_models):
            if p_values[i, j] < 0.05:
                print(f"{model_names[i]} vs {model_names[j]}: p={p_values[i, j]:.4f}, d={effect_sizes[i, j]:.2f}")
    
    return p_df, es_df

def plot_statistical_results(results, metric='f1'):
    """Visualize statistical comparison of models"""
    # Prepare data
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
    
    # Sort by performance
    sorted_idx = np.argsort(metric_means)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    metric_means = [metric_means[i] for i in sorted_idx]
    ci_lowers = [ci_lowers[i] for i in sorted_idx]
    ci_uppers = [ci_uppers[i] for i in sorted_idx]
    
    # Create plot
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
    
    # Add values to bars
    for i, v in enumerate(metric_means):
        plt.text(v + 0.01, i, f"{v:.3f}", color='black', va='center')
    
    plt.tight_layout()
    plt.savefig(f"results/{metric}_comparison_with_ci.png")
    plt.close()

# ================================================
# Main Execution Pipeline with Optimizations
# ================================================

def main():
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_and_combine_datasets()
    attack_df = preprocess_data(df)  # Get attack-only dataset
    
    # 2. Prepare data for attack-type classification
    X = attack_df.drop(columns=['attack_cat', 'attack_type'])
    y = attack_df['attack_type']
    
    # 3. Encode attack types
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42  # Smaller test size
    )
    
    # 5. Scale features
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    # 6. Attack-type classification
    print("\nRunning attack-type classification...")
    attack_results = train_and_evaluate(MODELS, X_train, y_train, X_test, y_test, 'multiclass')
    
    # 7. Advanced performance analysis
    print("\nPerforming statistical analysis...")
    perform_statistical_analysis(attack_results, 'f1')
    plot_statistical_results(attack_results, 'f1')
    
    # 8. Save comprehensive results
    results_data = []
    for model_name, metrics in attack_results.items():
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
    results_df.to_csv("results/attack_type_classification_results.csv", index=False)
    
    # 9. Generate detailed classification reports
    print("\nGenerating classification reports...")
    for model_name in MODELS.keys():
        try:
            model = joblib.load(f"models/{model_name.replace(' ', '_').lower()}_multiclass.pkl")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f"results/{model_name}_classification_report.csv")
        except:
            print(f"Could not generate report for {model_name}")
    
    print("\n=== Attack-Type Classification Complete ===")
    print(f"Classified {len(le.classes_)} attack types")
    print("Results saved to results/ directory")

if __name__ == "__main__":
    main()