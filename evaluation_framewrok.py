import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from config import ANALYSIS_DIR

class EvaluationFramework:
    """Handles statistical evaluation and result reporting"""
    def __init__(self, results):
        self.results = results
        self.models = list(results.keys())
        
    def generate_reports(self):
        """Generate comprehensive evaluation reports"""
        self.summary_statistics()
        self.paired_t_tests()
        self.metrics_comparison()
        
    def summary_statistics(self):
        """Generate summary statistics with confidence intervals"""
        summary = {}
        
        for model, data in self.results.items():
            cv_metrics = data['cv_metrics']
            model_summary = {}
            
            for metric, values in cv_metrics.items():
                mean = values['mean']
                std = values['std']
                ci_low, ci_high = stats.t.interval(
                    0.95, len(values['values'])-1, 
                    loc=mean, scale=std/np.sqrt(len(values['values']))
                )

                model_summary[metric] = {
                    'mean': mean,
                    'std': std,
                    'ci_95_low': ci_low,
                    'ci_95_high': ci_high
                }
            
            # Add test metrics
            model_summary['test_metrics'] = data['test_metrics']
            model_summary['inference_time'] = data['inference_time']
            model_summary['train_time'] = data['train_time']
            
            summary[model] = model_summary
        
        # Save summary to CSV
        summary_df = pd.DataFrame(summary).T
        summary_df.to_csv(f"{ANALYSIS_DIR}/summary_statistics.csv")
        
        return summary
    
    def paired_t_tests(self):
        """Perform paired t-tests between models"""
        t_test_results = {}
        
        # Compare all pairs of models
        for i in range(len(self.models)):
            for j in range(i+1, len(self.models)):
                model1 = self.models[i]
                model2 = self.models[j]
                
                comparison_key = f"{model1}_vs_{model2}"
                t_test_results[comparison_key] = {}
                
                # Compare for each metric
                for metric in self.results[model1]['cv_metrics'].keys():
                    values1 = self.results[model1]['cv_metrics'][metric]['values']
                    values2 = self.results[model2]['cv_metrics'][metric]['values']
                    
                    t_stat, p_value = stats.ttest_rel(values1, values2)
                    t_test_results[comparison_key][metric] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
        
        # Save t-test results to CSV
        t_test_df = pd.DataFrame(t_test_results).T
        t_test_df.to_csv(f"{ANALYSIS_DIR}/paired_t_tests.csv")
        
        # Visualize significance
        self.plot_significance_matrix(t_test_results)
        
        return t_test_results
    
    def plot_significance_matrix(self, t_test_results):
        """Create significance matrix visualization"""
        metrics = list(next(iter(t_test_results.values())).keys())
        significance_matrix = pd.DataFrame(
            index=self.models, columns=self.models)
        
        # Fill matrix with significance markers
        for comp, results in t_test_results.items():
            model1, model2 = comp.split('_vs_')
            significant = any(res['significant'] for res in results.values())
            significance_matrix.loc[model1, model2] = '★' if significant else '–'
            significance_matrix.loc[model2, model1] = '★' if significant else '–'
        
        # Plot matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            significance_matrix.fillna('').applymap(lambda x: 1 if x == '★' else 0),
            annot=significance_matrix.fillna(''),
            fmt='', cmap='coolwarm', cbar=False,
            linewidths=1, linecolor='black'
        )
        plt.title('Statistical Significance of Model Differences (★ = significant)')
        plt.savefig(f"{ANALYSIS_DIR}/significance_matrix.png")
        plt.close()
    
    def metrics_comparison(self):
        """Create visual comparison of model metrics"""
        # Prepare data
        metrics_data = []
        for model, data in self.results.items():
            for metric, values in data['test_metrics'].items():
                metrics_data.append({
                    'Model': model,
                    'Metric': metric.capitalize(),
                    'Value': values
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Value', hue='Model', data=df)
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{ANALYSIS_DIR}/metrics_comparison.png")
        plt.close()