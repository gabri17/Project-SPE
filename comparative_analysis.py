import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, f_oneway
import joblib

def load_feature_set_results(feature_folders):
    """Load results from all feature set folders into a combined DataFrame"""
    all_data = []
    
    for folder in feature_folders:
        feature_set = folder.split('_')[-1]
        
        results_file = os.path.join(folder, 'results', 'intrusion_detection_results.csv')
        if not os.path.exists(results_file):
            print(f"Warning: Results file not found in {folder}")
            continue
            
        df = pd.read_csv(results_file)
        df['Feature Set'] = feature_set
        
        reports = glob.glob(os.path.join(folder, 'results', '*_classification_report.csv'))
        for report in reports:
            model_name = os.path.basename(report).split('_')[0].replace('_', ' ')
            report_df = pd.read_csv(report, index_col=0)
            
            normal_precision = report_df.loc['Normal', 'precision']
            attack_precision = report_df.loc['Attack', 'precision']
            normal_recall = report_df.loc['Normal', 'recall']
            attack_recall = report_df.loc['Attack', 'recall']
            
            mask = (df['Model'] == model_name)
            if mask.any():
                idx = df[mask].index[0]
                df.loc[idx, 'Normal Precision'] = normal_precision
                df.loc[idx, 'Attack Precision'] = attack_precision
                df.loc[idx, 'Normal Recall'] = normal_recall
                df.loc[idx, 'Attack Recall'] = attack_recall
        
        all_data.append(df)
    
    return pd.concat(all_data, ignore_index=True)

def calculate_performance_differences(df):
    mean_perf = df.groupby('Model').agg({
        'Test Accuracy': 'mean',
        'Test Precision': 'mean',
        'Test Recall': 'mean',
        'Test F1': 'mean',
        'Test AUC': 'mean'
    }).reset_index()
    mean_perf.columns = ['Model'] + [f'Mean {col}' for col in mean_perf.columns[1:]]
    
    diff_perf = df.groupby('Model').agg({
        'Test Accuracy': lambda x: x.max() - x.min(),
        'Test Precision': lambda x: x.max() - x.min(),
        'Test Recall': lambda x: x.max() - x.min(),
        'Test F1': lambda x: x.max() - x.min(),
        'Test AUC': lambda x: x.max() - x.min()
    }).reset_index()
    diff_perf.columns = ['Model'] + [f'Δ {col}' for col in diff_perf.columns[1:]]
    
    std_perf = df.groupby('Model').agg({
        'Test Accuracy': 'std',
        'Test Precision': 'std',
        'Test Recall': 'std',
        'Test F1': 'std',
        'Test AUC': 'std'
    }).reset_index()
    std_perf.columns = ['Model'] + [f'Std {col}' for col in std_perf.columns[1:]]
    
    performance_df = mean_perf.merge(diff_perf, on='Model')
    performance_df = performance_df.merge(std_perf, on='Model')
    
    performance_df = performance_df.sort_values('Mean Test Accuracy', ascending=False)
    
    return performance_df

def compare_feature_sets(df, output_dir):

    anova_results = []
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        f_val, p_val = f_oneway(
            *[group['Test Accuracy'].values for _, group in model_df.groupby('Feature Set')]
        )
        anova_results.append({
            'Model': model,
            'F-value': f_val,
            'p-value': p_val,
            'Significant (p<0.05)': p_val < 0.05
        })
    
    anova_df = pd.DataFrame(anova_results)
    
    pairwise_results = []
    feature_sets = df['Feature Set'].unique()
    
    for i in range(len(feature_sets)):
        for j in range(i+1, len(feature_sets)):
            set1 = feature_sets[i]
            set2 = feature_sets[j]
            
            set1_acc = df[df['Feature Set'] == set1].groupby('Model')['Test Accuracy'].mean()
            set2_acc = df[df['Feature Set'] == set2].groupby('Model')['Test Accuracy'].mean()
            
            t_stat, p_val = ttest_rel(set1_acc, set2_acc)
            
            pairwise_results.append({
                'Feature Set 1': set1,
                'Feature Set 2': set2,
                't-statistic': t_stat,
                'p-value': p_val,
                'Significant (p<0.05)': p_val < 0.05,
                'Mean Accuracy Diff': (set1_acc - set2_acc).mean()
            })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    anova_df.to_csv(os.path.join(output_dir, 'anova_results.csv'), index=False)
    pairwise_df.to_csv(os.path.join(output_dir, 'pairwise_feature_set_comparison.csv'), index=False)
    
    return anova_df, pairwise_df

def visualize_results(df, output_dir):
    """Create visualizations comparing model performance across feature sets"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Model', y='Test Accuracy', hue='Feature Set')
    plt.title('Model Accuracy Across Feature Sets')
    plt.ylabel('Accuracy')
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=15)
    plt.legend(title='Feature Set')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
    plt.close()
    
    stability_df = df.groupby('Model').agg(
        Mean_Accuracy=('Test Accuracy', 'mean'),
        Std_Accuracy=('Test Accuracy', 'std')
    ).reset_index().sort_values('Mean_Accuracy', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stability_df, x='Mean_Accuracy', y='Std_Accuracy', hue='Model', s=200)
    plt.title('Model Performance Stability Across Feature Sets')
    plt.xlabel('Mean Accuracy')
    plt.ylabel('Standard Deviation (Accuracy)')
    plt.xlim(0.7, 1.0)
    
    for i, row in stability_df.iterrows():
        plt.text(row['Mean_Accuracy'] + 0.002, row['Std_Accuracy'] + 0.0002, 
                 row['Model'], fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_stability.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='Normal Recall', y='Normal Precision', 
                    hue='Model', style='Feature Set', s=150)
    plt.title('Precision-Recall Tradeoff for Normal Traffic Detection')
    plt.xlabel('Recall (Normal)')
    plt.ylabel('Precision (Normal)')
    plt.xlim(0.5, 1.0)
    plt.ylim(0.7, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_tradeoff_normal.png'))
    plt.close()
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='Model', y='Test AUC')
    plt.title('AUC Distribution Across Feature Sets')
    plt.ylabel('AUC Score')
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'))
    plt.close()
    
    metrics = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1', 'Test AUC']
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x='Model', y=metric, hue='Feature Set', ax=axes[i])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=15)
        axes[i].set_ylim(0.5, 1.0)
    
    fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_performance_comparison.png'))
    plt.close()

def main():
    feature_folders = [
        'output_selected_features_first',
        'output_selected_features_second',
        'output_selected_features_third',
        'output_selected_features_fourth'
    ]
    
    output_dir = 'comparative_analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading results from all feature sets...")
    combined_df = load_feature_set_results(feature_folders)
    combined_df.to_csv(os.path.join(output_dir, 'all_results_combined.csv'), index=False)
    
    print("Calculating performance differences...")
    performance_df = calculate_performance_differences(combined_df)
    performance_df.to_csv(os.path.join(output_dir, 'performance_summary.csv'), index=False)
    
    print("Comparing feature sets with statistical tests...")
    anova_df, pairwise_df = compare_feature_sets(combined_df, output_dir)
    
    print("Creating visualizations...")
    visualize_results(combined_df, output_dir)
    
    print("Analysis complete! Results saved to:", output_dir)

if __name__ == "__main__":
    main()