import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel
import joblib
from sklearn.metrics import classification_report, RocCurveDisplay

def calculate_confidence_interval(scores, confidence=0.95):
    mean = np.mean(scores)
    std_err = stats.sem(scores)
    h = std_err * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
    return mean, mean - h, mean + h

def perform_statistical_analysis(results, metric='f1'):    
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
    
    for i, (model_name, data) in enumerate(results.items(), 1):
        cm = data['confusion_matrix']
        
        cm_df = pd.DataFrame(
            cm,
            index=['Normal', 'Attack'], 
            columns=['Normal', 'Attack'] 
        )
        cm_df.to_csv(f"results/{model_name}_confusion_matrix.csv")
        
        plt.subplot(2, 3, i)
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig("results/confusion_matrices_binary.png")
    plt.close()

def generate_reports(models, X_test, y_test):
    print("\nGenerating classification reports...")
    for model_name in models.keys():
        try:
            model = joblib.load(f"models/{model_name.replace(' ', '_').lower()}_binary.pkl")
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, target_names=['Normal', 'Attack'], output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_df.to_csv(f"results/{model_name}_classification_report.csv")
            
            if hasattr(model, 'predict_proba'):
                RocCurveDisplay.from_predictions(y_test, model.predict_proba(X_test)[:, 1])
                plt.title(f'ROC Curve - {model_name}')
                plt.savefig(f"results/{model_name}_roc_curve.png")
                plt.close()
        except Exception as e:
            print(f"Could not generate report for {model_name}: {str(e)}")

def plot_accuracy_comparison(results):
    model_names = []
    test_accuracies = []
    cv_acc_means = []
    cv_acc_stds = []
    
    for model_name, data in results.items():
        model_names.append(model_name)
        test_accuracies.append(data['accuracy'])
        cv_acc_means.append(np.mean(data['cv_scores']['accuracy']))
        cv_acc_stds.append(np.std(data['cv_scores']['accuracy']))
    
    # Sort by test accuracy
    sorted_idx = np.argsort(test_accuracies)[::-1]
    model_names = [model_names[i] for i in sorted_idx]
    test_accuracies = [test_accuracies[i] for i in sorted_idx]
    cv_acc_means = [cv_acc_means[i] for i in sorted_idx]
    cv_acc_stds = [cv_acc_stds[i] for i in sorted_idx]
    
    # Create plot
    plt.figure(figsize=(14, 8))
    x_pos = np.arange(len(model_names))
    
    # Test accuracy bars
    plt.bar(x_pos, test_accuracies, width=0.4, 
            label='Test Accuracy', color='skyblue')
    
    # CV accuracy points with error bars
    plt.errorbar(x_pos, cv_acc_means, yerr=cv_acc_stds, 
                 fmt='o', color='darkred', capsize=5,
                 label='CV Accuracy (Mean ± SD)')
    
    # Add values
    for i, v in enumerate(test_accuracies):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    for i, (mean, std) in enumerate(zip(cv_acc_means, cv_acc_stds)):
        plt.text(i, mean + 0.01, f"{mean:.3f}±{std:.3f}", 
                 ha='center', color='darkred')
    
    plt.xticks(x_pos, model_names, rotation=15)
    plt.ylabel('Accuracy Score')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig("results/accuracy_comparison.png")
    plt.close()