import os
import numpy as np
import joblib
import pandas as pd
from data_loading import load_and_combine_datasets, preprocess_data, scale_features
from models import MODELS, train_and_evaluate
from analysis import perform_statistical_analysis, plot_statistical_results, plot_confusion_matrices, generate_reports

def main():
    for d in ("models", "results"):
        os.makedirs(d, exist_ok=True)
    
    print("Loading and preprocessing data...")
    processed_df = preprocess_data(load_and_combine_datasets())

    train_df = processed_df[processed_df['dataset'] == 'train']
    test_df = processed_df[processed_df['dataset'] == 'test']
    
    train_df = train_df.drop(columns=['dataset'])
    test_df = test_df.drop(columns=['dataset'])
    
    X_train = train_df.drop(columns=['attack_cat', 'is_attack', 'label'])
    y_train = train_df['is_attack']
    X_test = test_df.drop(columns=['attack_cat', 'is_attack', 'label'])
    y_test = test_df['is_attack']
    
    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)
    
    print("\nRunning intrusion detection...")
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
    
    generate_reports(MODELS, X_test, y_test)
    
    print("\nIntrusion Detection Complete")
    print(f"Processed {len(processed_df)} records ({y_test.sum()} attacks, {len(y_test)-y_test.sum()} normal)")
    print("Results saved to results/ directory")

if __name__ == "__main__":
    main()