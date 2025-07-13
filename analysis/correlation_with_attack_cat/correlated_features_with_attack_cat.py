from data_loader import UNSWNB15Loader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import f_oneway

loader = UNSWNB15Loader()
df = loader.load()

df['attack_cat'] = df['attack_cat'].fillna('Normal')
df['attack_cat'] = df['attack_cat'].replace('', 'Normal')

df['attack_cat'] = df['attack_cat'].str.strip() 
df['attack_cat'] = df['attack_cat'].replace('Backdoors', 'Backdoor')

df = df[df['attack_cat'] != 'Normal'] #we just want to analyze the attack categories

print("Unique values in 'attack_cat':", df['attack_cat'].unique())

print("Dataset loaded successfully.")

numeric_cols = df.select_dtypes(include='number').columns.drop(['Stime', 'Ltime'])

def eta_squared(groups):
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum([len(g)*(np.mean(g)-grand_mean)**2 for g in groups])
    ss_total = sum((all_data - grand_mean)**2)
    return ss_between / ss_total

print(f"\ANOVA results for numeric Features by attack category:")

for num in numeric_cols:
    groups = [group[num].values for name, group in df.groupby('attack_cat')]
    if len(groups) > 1:
        stat, p = f_oneway(*groups)
        eta2 = eta_squared(groups)
        print(f"{num}: p-value={p:.4g}, eta squared={eta2:.4f}")

with open('analysis/correlation_with_attack_cat/anova_results.txt', 'w') as f:
    f.write("ANOVA results for numeric Features by attack category:\n")
    for num in numeric_cols:
        groups = [group[num].values for name, group in df.groupby('attack_cat')]
        if len(groups) > 1:
            stat, p = f_oneway(*groups)
            eta2 = eta_squared(groups)
            f.write(f"{num}: p-value={p:.4g}, eta squared={eta2:.4f}\n")