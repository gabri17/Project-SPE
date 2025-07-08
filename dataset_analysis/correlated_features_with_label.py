from data_loader import UNSWNB15Loader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import numpy as np

loader = UNSWNB15Loader()
df = loader.load()

print("Dataset loaded successfully.")

#1) compute correlations between numerical features and label
label_correlations = df.corr(numeric_only=True)['label'].drop('label').sort_values(ascending=False)

TO_CONSIDER = 10

print(f"Top {TO_CONSIDER} positive correlations with label:")
print(label_correlations.head(TO_CONSIDER))

print(f"\nTop {TO_CONSIDER} negative correlations with label:")
print(label_correlations.tail(TO_CONSIDER))

with open("dataset_analysis/results/correlation_results.txt", "w") as f:
    f.write(f"\n\nTop {TO_CONSIDER} positive correlations with label:\n")
    f.write(label_correlations.head(TO_CONSIDER).to_string())
    f.write(f"\n\nTop {TO_CONSIDER} negative correlations with label:\n")
    f.write(label_correlations.tail(TO_CONSIDER).to_string())
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    f.write(f"\n\nCategorical columns: {categorical_cols.tolist()}\n")


#2) analysis of categorical features respect to label
useful_categorical_cols = ['state', 'service', 'ct_ftp_cmd']

if(False):
    for col in categorical_cols:
        print(f"Distribution of '{col}' respect to label:")
        print(pd.crosstab(df[col], df['label']))
        #bar plot
        pd.crosstab(df[col], df['label']).plot(kind='bar', stacked=True)
        plt.title(f"Distribution of {col} respect to label")
        plt.ylabel("Count")
        plt.savefig(f"dataset_analysis/results/distribution_{col}_respect_to_label.png")

#3) distribution of numerical features for each class of the label
numeric_cols = df.select_dtypes(include='number').columns.drop('label')
for col in label_correlations.index.tolist()[:5]:  #limiting to 5 features for example
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='label', y=col, data=df)
    plt.title(f"Distribution of {col} for label")
    plt.savefig(f"dataset_analysis/results/distribution_{col}_per_label.png")

#4) analysis of outliers in numerical features

#logarithmic transformation (add 1 to avoid log(0))
feature = 'ct_src_dport_ltm'
df[feature] = np.log1p(df[feature])

#separate outlier analysis (using boxplot whisker rule)
def find_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series[(series < lower) | (series > upper)]

outliers_0 = find_outliers(df[df['label'] == 0][feature])
outliers_1 = find_outliers(df[df['label'] == 1][feature])

print(f"Number of outliers for label=0: {len(outliers_0)}")
print(f"Number of outliers for label=1: {len(outliers_1)}")

#outlier statistics
print("Outlier statistics for label=0:")
print(outliers_0.describe())
print("\nOutlier statistics for label=1:")
print(outliers_1.describe())