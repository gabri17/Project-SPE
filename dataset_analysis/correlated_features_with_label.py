from data_loader import UNSWNB15Loader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

loader = UNSWNB15Loader()
df = loader.load()

print("Dataset loaded successfully.")

#1) compute correlations between numerical features and label
label_correlations = df.corr(numeric_only=True)['label'].drop('label').sort_values(ascending=False)

TO_CONSIDER = 20

print(f"Top {TO_CONSIDER} positive correlations with label:")
print(label_correlations.head(TO_CONSIDER))

print(f"\nTop {TO_CONSIDER} negative correlations with label:")
print(label_correlations.tail(TO_CONSIDER))

with open("dataset_analysis/results/corr_with_label/correlation_results.txt", "w") as f:
    f.write(f"\n\nTop {TO_CONSIDER} positive correlations with label:\n")
    f.write(label_correlations.head(TO_CONSIDER).to_string())
    f.write(f"\n\nTop {TO_CONSIDER} negative correlations with label:\n")
    f.write(label_correlations.tail(TO_CONSIDER).to_string())
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    f.write(f"\n\nCategorical columns: {categorical_cols.tolist()}\n")


#2) analysis of categorical features respect to label
useful_categorical_cols = ['state', 'service', 'ct_ftp_cmd']

useful_categorical_cols = label_correlations.index.tolist()[:10]

if(True):
    for col in useful_categorical_cols:
        print(f"Distribution of '{col}' respect to label:")
        print(pd.crosstab(df[col], df['label']))

        with open("dataset_analysis/results/corr_with_label/correlation_results.txt", "a") as f:
            f.write(f"\nDistribution of '{col}' respect to label:\n")
            f.write(pd.crosstab(df[col], df['label']).to_string())

        #bar plot
        pd.crosstab(df[col], df['label']).plot(kind='bar', stacked=True)
        plt.title(f"Distribution of {col} respect to label")
        plt.ylabel("Count")
        plt.savefig(f"dataset_analysis/results/corr_with_label/distribution_{col}_bar_plot_to_label.png")

#3) distribution of numerical features for each class of the label
numeric_cols = df.select_dtypes(include='number').columns.drop('label')
for col in label_correlations.index.tolist()[:10]:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='label', y=col, data=df)
    plt.title(f"Distribution of {col} for label")
    plt.savefig(f"dataset_analysis/results/corr_with_label/distribution_{col}_boxplot_to_label.png")

#4) analysis of outliers in numerical features
def outlier_analysis(df, feature, label_col='label'):
    """
    Performs outlier analysis for a given feature, separated by label.
    Applies log1p transformation to the feature.
    Prints number and statistics of outliers for each label.
    """
    # Logarithmic transformation
    df = df.copy()
    df[feature] = np.log1p(df[feature])

    def find_outliers(series):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return series[(series < lower) | (series > upper)]

    for label in df[label_col].unique():
        group = df[df[label_col] == label][feature]
        outliers = find_outliers(group)
        print(f"Number of outliers for {label_col}={label}: {len(outliers)}")
        print(f"Outlier statistics for {label_col}={label}:")
        print(outliers.describe())
        print()
# Example usage:
#outlier_analysis(df, 'ct_src_dport_ltm', label_col='label')