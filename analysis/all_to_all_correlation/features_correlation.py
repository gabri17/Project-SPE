import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import UNSWNB15Loader

loader = UNSWNB15Loader()
df = loader.load()

##Correlation analysis
corr_matrix = df.corr(numeric_only=True)

corr_unstacked = corr_matrix.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked < 1]  #avoid self-correlation
corr_sorted = corr_unstacked.sort_values(ascending=False).drop_duplicates()

THRESHOLD = 0.8

high_corr = corr_sorted[corr_sorted > THRESHOLD]

print(f"Pair of features with high correlations (> {THRESHOLD}):")
print(high_corr)

#heatmap of correlations among different features
plt.figure(figsize=(18, 16))
ax = sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    cbar_kws={'shrink': 0.8, 'label': 'Correlation', 'format': '%.2f'}
)
plt.title("Heatmap of correlations between numerical features", fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=16)
plt.yticks(rotation=0, fontsize=16)
ax.set_xlabel("Feature", fontsize=18)
ax.set_ylabel("Feature", fontsize=18)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation', fontsize=18)
plt.tight_layout()
plt.savefig("analysis/all_to_all_correlation/heatmap_correlations.png")
plt.close()

#summary table of most correlated features
summary = []
for col in corr_matrix.columns:
    correlated = corr_matrix[col][(corr_matrix[col].abs() > THRESHOLD) & (corr_matrix[col].abs() < 1)].index.tolist()
    if correlated:
        summary.append({'feature': col, 'correlated_with': ', '.join(correlated)})

summary_df = pd.DataFrame(summary)
print(f"\nSummary table of features with correlation > {THRESHOLD}:")
print(summary_df)

with open("analysis/all_to_all_correlation/summary_features.txt", "w") as f:
    f.write(f"Pair of features with high correlations (> {THRESHOLD}):\n")
    f.write(high_corr.to_string())
    f.write(f"\nSummary table of features with correlation > {THRESHOLD}:\n")
    f.write(summary_df.to_string(index=False))