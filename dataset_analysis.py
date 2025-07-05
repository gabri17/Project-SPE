import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import ANALYSIS_DIR

def analyze_dataset(df_train, df_test):
    """Perform comprehensive dataset analysis"""
    # Basic dataset stats
    analysis_results = {}
    analysis_results["train_shape"] = df_train.shape
    analysis_results["test_shape"] = df_test.shape
    analysis_results["train_missing_values"] = df_train.isnull().sum().sum()
    analysis_results["test_missing_values"] = df_test.isnull().sum().sum()
    
    # Class distribution analysis
    class_dist = {
        "binary_train": df_train['label'].value_counts(normalize=True).to_dict(),
        "binary_test": df_test['label'].value_counts(normalize=True).to_dict(),
        "multiclass_train": df_train['attack_cat'].value_counts(normalize=True).to_dict(),
        "multiclass_test": df_test['attack_cat'].value_counts(normalize=True).to_dict()
    }
    
    # Visualization: Class distributions
    fig, ax = plt.subplots(2, 2, figsize=(16, 14))
    
    # Binary class distribution
    df_train['label'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0, 0])
    ax[0, 0].set_title('Training Set: Binary Class Distribution')
    df_test['label'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0, 1])
    ax[0, 1].set_title('Testing Set: Binary Class Distribution')
    
    # Multiclass distribution
    df_train['attack_cat'].value_counts().plot.bar(ax=ax[1, 0])
    ax[1, 0].set_title('Training Set: Attack Category Distribution')
    ax[1, 0].tick_params(axis='x', rotation=45)
    df_test['attack_cat'].value_counts().plot.bar(ax=ax[1, 1])
    ax[1, 1].set_title('Testing Set: Attack Category Distribution')
    ax[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{ANALYSIS_DIR}/class_distributions.png")
    plt.close()
    
    # Feature correlation analysis
    corr_matrix = df_train.corr(numeric_only=True)
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.savefig(f"{ANALYSIS_DIR}/feature_correlation.png")
    plt.close()
    
    # Top correlated features with targets
    binary_corr = corr_matrix['label'].abs().sort_values(ascending=False).head(10)
    multiclass_corr = corr_matrix['attack_cat_num'].abs().sort_values(ascending=False).head(10)
    
    binary_corr.to_csv(f"{ANALYSIS_DIR}/top_binary_correlations.csv")
    multiclass_corr.to_csv(f"{ANALYSIS_DIR}/top_multiclass_correlations.csv")
    
    # Generate feature distribution plots
    plot_feature_distributions(df_train, ANALYSIS_DIR)
    
    return {
        "dataset_stats": analysis_results,
        "class_distribution": class_dist,
        "top_binary_correlations": binary_corr.to_dict(),
        "top_multiclass_correlations": multiclass_corr.to_dict()
    }

def plot_feature_distributions(df, output_dir):
    """Plot distributions of key features"""
    # Select top 10 features for visualization
    top_features = df.corr(numeric_only=True)['label'].abs().sort_values(ascending=False).head(10).index
    
    plt.figure(figsize=(15, 20))
    for i, feature in enumerate(top_features, 1):
        plt.subplot(5, 2, i)
        sns.histplot(data=df, x=feature, hue='label', kde=True, element='step', stat='density', common_norm=False)
        plt.title(f'Distribution of {feature} by Label')
        plt.xlabel(feature)
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png")
    plt.close()