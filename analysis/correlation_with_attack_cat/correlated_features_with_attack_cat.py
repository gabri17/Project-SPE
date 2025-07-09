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
#df['attack_cat'] = pd.factorize(df['attack_cat'])[0] #we encode categorical variable as numerical variable
print("Unique values in 'attack_cat':", df['attack_cat'].unique())

print("Dataset loaded successfully.")

#1) distribution of numerical features for each class of the attack_cat
numeric_cols = df.select_dtypes(include='number').columns.drop(['Stime', 'Ltime'])

for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x='attack_cat', y=col, data=df)
    plt.title(f"Distribution of {col} for attack_cat")
    plt.savefig(f"analysis/correlation_with_attack_cat/distribution_{col}_per_attack_cat.png")

#2) t-test for verifying when we have statistically significant results
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

categorical_cols = df.select_dtypes(include='object').columns.drop('attack_cat')

for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    sns.countplot(x=col, hue='attack_cat', data=df)
    plt.title(f"Count of {col} for each attack_cat")
    plt.savefig(f"analysis/correlation_with_attack_cat/count_{col}_per_attack_cat.png")