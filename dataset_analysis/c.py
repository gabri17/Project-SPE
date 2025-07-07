import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
import numpy as np

#correlations between features and correlations with attack category and label

base_path = os.path.join("datasets", "CSV_Files")
additional_path = os.path.join('training_testing_sets')

file_1 = 'UNSW-NB15_1.csv'
file_2 = 'UNSW-NB15_2.csv'
file_3 = 'UNSW-NB15_3.csv'
file_4 = 'UNSW-NB15_4.csv'
file_training = 'UNSW_NB15_training-set.csv'
file_testing = ''

df = pd.read_csv(os.path.join(base_path, additional_path, file_training))
df['attack_cat_num'] = pd.factorize(df['attack_cat'])[0] #we encode categorical variable as numerical variable

toCompare = 'attack_cat_num'

##Correlation analysis
corr_matrix = df.corr(numeric_only=True)

corr_unstacked = corr_matrix.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1]  #avoid self-correlation
corr_sorted = corr_unstacked.sort_values(ascending=False).drop_duplicates()

high_corr = corr_sorted[corr_sorted > 0.9]

print("Pair of features with high correlations ( > 0.9):")
print(high_corr)

#heatmap of correlations among different features
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Heatmap of correlations between numerical features")
plt.show()

#dendrogram of features based on correlation
plt.figure(figsize=(12, 5))
corr_dist = 1 - np.abs(corr_matrix)
linkage_matrix = linkage(corr_dist, method='average')
dendrogram(linkage_matrix, labels=corr_matrix.columns, leaf_rotation=90)
plt.title("Dendrogram of features based on correlation")
plt.tight_layout()
plt.show()

#summary table of most correlated features
summary = []
threshold = 0.85  #correlation threshold
for col in corr_matrix.columns:
    correlated = corr_matrix[col][(corr_matrix[col].abs() > threshold) & (corr_matrix[col].abs() < 1)].index.tolist()
    if correlated:
        summary.append({'feature': col, 'correlated_with': ', '.join(correlated)})

summary_df = pd.DataFrame(summary)
print(f"\nSummary table of features with correlation > {threshold}:")
print(summary_df)