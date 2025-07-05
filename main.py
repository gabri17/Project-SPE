from data_loading import load_datasets
from dataset_analysis import analyze_dataset
from preprocessing import DataPreprocessor
from model_training import ModelTrainer
from evaluation_framewrok import EvaluationFramework
from feature_analysis import FeatureAnalyzer
import pandas as pd
import os
import numpy as np
from config import ANALYSIS_DIR
import matplotlib.pyplot as plt

def main():
    # Create analysis directory
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    
    # Step 1: Load and analyze data
    print("="*80)
    print("Loading and preparing datasets...")
    print("="*80)
    df_train, df_test = load_datasets()
    
    print("\n" + "="*80)
    print("Analyzing dataset...")
    print("="*80)
    analysis_results = analyze_dataset(df_train, df_test)
    print("Dataset analysis completed. Results saved to:", ANALYSIS_DIR)
    
    # Step 2: Preprocess data
    print("\n" + "="*80)
    print("Preprocessing data...")
    print("="*80)
    preprocessor = DataPreprocessor()
    
    X_train, X_test, y_train_bin, y_test_bin, y_train_multi, y_test_multi = preprocessor.prepare_data(
        df_train, df_test
    )
    
    feature_names = preprocessor.feature_names
    print(f"Preprocessing completed. Feature count: {len(feature_names)}")
    
    # Step 3: Feature analysis
    print("\n" + "="*80)
    print("Analyzing feature correlations...")
    print("="*80)
    feature_analyzer = FeatureAnalyzer(X_train, feature_names)
    correlation_report = feature_analyzer.analyze_correlations()
    print(f"Found {len(correlation_report)} highly correlated feature pairs")
    
    # Step 4: Train models with all features
    print("\n" + "="*80)
    print("Training models for binary classification...")
    print("="*80)
    bin_trainer = ModelTrainer(X_train, X_test, y_train_bin, y_test_bin, feature_names, 'binary')
    bin_results = bin_trainer.train_models()
    
    print("\n" + "="*80)
    print("Training models for multiclass classification...")
    print("="*80)
    multi_trainer = ModelTrainer(X_train, X_test, y_train_multi, y_test_multi, feature_names, 'multiclass')
    multi_results = multi_trainer.train_models()
    
    # Step 5: Statistical evaluation
    print("\n" + "="*80)
    print("Evaluating binary classification results...")
    print("="*80)
    bin_evaluator = EvaluationFramework(bin_results)
    bin_evaluation = bin_evaluator.generate_reports()
    
    print("\n" + "="*80)
    print("Evaluating multiclass classification results...")
    print("="*80)
    multi_evaluator = EvaluationFramework(multi_results)
    multi_evaluation = multi_evaluator.generate_reports()
    
    # Step 6: Redo training with reduced feature set
    print("\n" + "="*80)
    print("Removing correlated features and retraining...")
    print("="*80)
    X_train_reduced, reduced_features = feature_analyzer.remove_correlated_features()
    X_test_reduced = pd.DataFrame(X_test, columns=feature_names)[reduced_features].values
    
    print(f"\nTraining binary models with reduced features ({len(reduced_features)} features)...")
    bin_trainer_reduced = ModelTrainer(X_train_reduced, X_test_reduced, y_train_bin, y_test_bin, reduced_features, 'binary')
    bin_results_reduced = bin_trainer_reduced.train_models()
    
    print(f"\nTraining multiclass models with reduced features ({len(reduced_features)} features)...")
    multi_trainer_reduced = ModelTrainer(X_train_reduced, X_test_reduced, y_train_multi, y_test_multi, reduced_features, 'multiclass')
    multi_results_reduced = multi_trainer_reduced.train_models()
    
    # Step 7: Compare full vs reduced feature sets
    print("\n" + "="*80)
    print("Comparing full vs reduced feature set performance...")
    print("="*80)
    bin_comparison = {}
    for model in bin_results:
        bin_comparison[model] = {
            'full_features': bin_results[model]['test_metrics']['f1'],
            'reduced_features': bin_results_reduced[model]['test_metrics']['f1']
        }
    
    multi_comparison = {}
    for model in multi_results:
        multi_comparison[model] = {
            'full_features': multi_results[model]['test_metrics']['f1'],
            'reduced_features': multi_results_reduced[model]['test_metrics']['f1']
        }
    
    # Save comparisons
    bin_comp_df = pd.DataFrame(bin_comparison).T
    bin_comp_df.to_csv(f"{ANALYSIS_DIR}/binary_feature_comparison.csv")
    
    multi_comp_df = pd.DataFrame(multi_comparison).T
    multi_comp_df.to_csv(f"{ANALYSIS_DIR}/multiclass_feature_comparison.csv")
    
    # Generate comparison plots
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    bin_comp_df.plot(kind='bar', rot=45)
    plt.title('Binary Classification: Full vs Reduced Features')
    plt.ylabel('F1 Score')
    plt.legend(title='Feature Set')
    
    plt.subplot(1, 2, 2)
    multi_comp_df.plot(kind='bar', rot=45)
    plt.title('Multiclass Classification: Full vs Reduced Features')
    plt.ylabel('Macro F1 Score')
    plt.legend(title='Feature Set')
    
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_DIR}/feature_set_comparison.png")
    plt.close()
    
    print("\n" + "="*80)
    print("Project execution completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()