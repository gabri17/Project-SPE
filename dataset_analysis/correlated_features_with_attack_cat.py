from data_loader import UNSWNB15Loader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from scipy.stats import f_oneway

loader = UNSWNB15Loader()
df = loader.load()

df['attack_cat'] = df['attack_cat'].fillna('Normal')
df['attack_cat'] = df['attack_cat'].replace('', 'Normal')

df['attack_cat'] = df['attack_cat'].str.strip() 
df['attack_cat'] = df['attack_cat'].replace('Backdoors', 'Backdoor')

print("Unique values in 'attack_cat':", df['attack_cat'].unique())
df['attack_cat'] = pd.factorize(df['attack_cat'])[0] #we encode categorical variable as numerical variable
print("Unique values in 'attack_cat':", df['attack_cat'].unique())

print("Dataset loaded successfully.")

#1) compute correlations between numerical features and attack_cat
attack_category_correlations = df.corr(numeric_only=True)['attack_cat'].drop(['attack_cat', 'label']).sort_values(ascending=False)

TO_CONSIDER = 10

print(f"Top {TO_CONSIDER} positive correlations with attack_cat:")
print(attack_category_correlations.head(TO_CONSIDER))

print(f"\nTop {TO_CONSIDER} negative correlations with attack_cat:")
print(attack_category_correlations.tail(TO_CONSIDER))

with open("dataset_analysis/results/correlation_results_with_attack.txt", "w") as f:
    f.write(f"\n\nTop {TO_CONSIDER} positive correlations with attack_cat:\n")
    f.write(attack_category_correlations.head(TO_CONSIDER).to_string())
    f.write(f"\n\nTop {TO_CONSIDER} negative correlations with attack_cat:\n")
    f.write(attack_category_correlations.tail(TO_CONSIDER).to_string())
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    f.write(f"\n\nCategorical columns: {categorical_cols.tolist()}\n")


#2) analysis of categorical features respect to attack_cat
useful_categorical_cols = ['state', 'service', 'ct_ftp_cmd']

if(False):
    for col in categorical_cols:
        print(f"Distribution of '{col}' respect to attack_cat:")
        print(pd.crosstab(df[col], df['attack_cat']))
        #bar plot
        pd.crosstab(df[col], df['attack_cat']).plot(kind='bar', stacked=True)
        plt.title(f"Distribution of {col} respect to attack_cat")
        plt.ylabel("Count")
        plt.savefig(f"dataset_analysis/results/distribution_{col}_respect_to_attack_cat.png")

#3) distribution of numerical features for each class of the attack_cat
numeric_cols = df.select_dtypes(include='number').columns.drop('attack_cat')
for col in attack_category_correlations.index.tolist()[0:5]:  #limiting to 5 features for example
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='attack_cat', y=col, data=df)
    plt.title(f"Distribution of {col} for attack_cat")
    plt.savefig(f"dataset_analysis/results/distribution_{col}_per_attack_cat.png")

#4) t-test for verifying when we have statistically significant results
print(f"\nAnalisi ANOVA per la feature categorica attack_cat")
def eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum([len(g)*(np.mean(g)-grand_mean)**2 for g in groups])
    ss_total = sum((all_data - grand_mean)**2)
    return ss_between / ss_total

for num in numeric_cols:
    groups = [group[num].values for name, group in df.groupby('attack_cat')]
    if len(groups) > 1:
        stat, p = f_oneway(*groups)
        eta2 = eta_squared(groups)
        print(f"{num}: p-value={p:.4g}, eta squared={eta2:.4f}")