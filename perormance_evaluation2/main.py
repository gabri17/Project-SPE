# main.py
import os
import numpy as np
import joblib
import pandas as pd
import hashlib
from data_loading import load_and_combine_datasets, preprocess_data, scale_features
from models import MODELS, train_and_evaluate
from analysis import perform_statistical_analysis, plot_accuracy_comparison, plot_statistical_results, plot_confusion_matrices, generate_reports
from perormance_evaluation2.config import FEATURE_SETS  

def main(feature_set_name="selected_features"):
    # Validate feature set
    if feature_set_name not in FEATURE_SETS:
        raise ValueError(f"Invalid feature set: {feature_set_name}")
    
    selected_features = FEATURE_SETS[feature_set_name]
    
    # Create unique output directory
    feature_hash = hashlib.md5(str(selected_features).encode()).hexdigest()[:8]
    base_dir = f"output_{feature_set_name}_{feature_hash}"
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    
    for d in (base_dir, models_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    print("Loading and preprocessing data...")
    processed_df = preprocess_data(load_and_combine_datasets())

    train_df = processed_df[processed_df['dataset'] == 'train']
    test_df = processed_df[processed_df['dataset'] == 'test']
    
    # Feature selection
    if selected_features:
        X_train = train_df[selected_features]
        X_test = test_df[selected_features]
    else:  # Use all features
        X_train = train_df.drop(columns=['attack_cat', 'is_attack', 'label'])
        X_test = test_df.drop(columns=['attack_cat', 'is_attack', 'label'])
    
    y_train = train_df['is_attack']
    y_test = test_df['is_attack']
    
    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    print(f"\nRunning intrusion detection with {feature_set_name}...")
    trained_models, binary_results = train_and_evaluate(
        MODELS, X_train, y_train, X_test, y_test, 'binary', 
        models_dir, results_dir
    )
    
    print("\nPerforming statistical analysis...")
    perform_statistical_analysis(binary_results, 'f1', results_dir)
    plot_statistical_results(binary_results, 'f1', results_dir)
    plot_confusion_matrices(binary_results, y_test, results_dir)
    plot_accuracy_comparison(binary_results, results_dir)
    perform_statistical_analysis(binary_results, 'accuracy', results_dir)
    plot_statistical_results(binary_results, 'accuracy', results_dir)

    
    results_data = []
    for model_name, metrics in binary_results.items():
        row = {
        'Model': model_name,
        'Feature Set': feature_set_name,
        'Train Time (s)': metrics.get('train_time', 'N/A'),
        'Test Accuracy': metrics['accuracy'],
        'Test Precision': metrics['precision'],
        'Test Recall': metrics['recall'],
        'Test F1': metrics['f1'],
        'Test AUC': metrics.get('roc_auc', 'N/A'),
        'CV Accuracy Mean': np.mean(metrics['cv_scores']['accuracy']),
        'CV Accuracy Std': np.std(metrics['cv_scores']['accuracy']),
        'CV F1 Mean': np.mean(metrics['cv_scores']['f1']),
        'CV F1 Std': np.std(metrics['cv_scores']['f1'])
        }
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(results_dir, "intrusion_detection_results.csv"), index=False)
    
    generate_reports(MODELS, X_test, y_test, models_dir, results_dir)
    
    print("\nIntrusion Detection Complete")
    print(f"Feature Set: {feature_set_name}")
    print(f"Processed {len(processed_df)} records ({y_test.sum()} attacks, {len(y_test)-y_test.sum()} normal)")
    print(f"Results saved to {base_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-set", default="selected_features",
                        help="Feature set to use (from config.py)")
    args = parser.parse_args()
    
    main(feature_set_name=args.feature_set)