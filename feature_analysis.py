import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import ANALYSIS_DIR, CORRELATION_THRESHOLD

class FeatureAnalyzer:
    """Handles feature analysis and correlation management"""
    def __init__(self, X, feature_names):
        self.X = X
        self.feature_names = feature_names
        self.df = pd.DataFrame(X, columns=feature_names)
        
    def analyze_correlations(self):
        """Analyze feature correlations and return report"""
        # Calculate correlation matrix
        corr_matrix = self.df.corr().abs()
        
        # Plot correlation heatmap
        plt.figure(figsize=(20, 18))
        sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlation Matrix')
        plt.savefig(f"{ANALYSIS_DIR}/feature_correlation_matrix.png")
        plt.close()
        
        # Identify highly correlated features
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_pairs = [(i, j) for i in range(len(corr_matrix.columns)) 
                          for j in range(i+1, len(corr_matrix.columns)) 
                          if corr_matrix.iloc[i, j] > CORRELATION_THRESHOLD]
        
        # Create correlation report
        correlation_report = []
        for i, j in correlated_pairs:
            feature1 = corr_matrix.columns[i]
            feature2 = corr_matrix.columns[j]
            correlation_report.append({
                'feature1': feature1,
                'feature2': feature2,
                'correlation': corr_matrix.iloc[i, j]
            })
        
        # Save correlation report
        corr_df = pd.DataFrame(correlation_report)
        corr_df.to_csv(f"{ANALYSIS_DIR}/high_correlation_features.csv", index=False)
        
        # Plot top correlations if any exist
        if not corr_df.empty:
            plt.figure(figsize=(12, 8))
            # Create a combined label for better visualization
            corr_df['feature_pair'] = corr_df['feature1'] + ' & ' + corr_df['feature2']
            top_corr = corr_df.nlargest(20, 'correlation')
            
            # Plot horizontal bar chart
            plt.barh(top_corr['feature_pair'], top_corr['correlation'])
            plt.title('Top Feature Correlations')
            plt.xlabel('Correlation Coefficient')
            plt.ylabel('Feature Pairs')
            plt.tight_layout()
            plt.savefig(f"{ANALYSIS_DIR}/top_correlations.png")
            plt.close()
        
        # Return the correlation report to the caller
        return correlation_report

    def remove_correlated_features(self):
        """Remove correlated features based on threshold"""
        # Reuse the same logic from analyze_correlations
        corr_matrix = self.df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to remove
        to_drop = [column for column in upper.columns 
                   if any(upper[column] > CORRELATION_THRESHOLD)]
        
        # Get list of features to keep
        features_to_keep = [col for col in self.df.columns if col not in to_drop]
        
        # Return reduced dataset and feature list
        return self.df[features_to_keep].values, features_to_keep