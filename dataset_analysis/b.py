import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import numpy as np
import random
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.preprocessing import MinMaxScaler
#correlations between features and correlations with attack category and label

base_path = os.path.join("datasets", "CSV_Files")
additional_path = os.path.join('training_testing_sets')

file_1 = 'UNSW-NB15_1.csv'
file_2 = 'UNSW-NB15_2.csv'
file_3 = 'UNSW-NB15_3.csv'
file_4 = 'UNSW-NB15_4.csv'
file_training = 'UNSW_NB15_training-set.csv'
file_testing = ''
name_features_file = 'NUSW-NB15_features.csv'

names = pd.read_csv(os.path.join(base_path, name_features_file), encoding='cp1252')['Name'].str.strip()

df1 = pd.read_csv(os.path.join(base_path, file_1), header=None, low_memory=False)
df2 = pd.read_csv(os.path.join(base_path, file_2), header=None, low_memory=False)
df3 = pd.read_csv(os.path.join(base_path, file_3), header=None, low_memory=False)
df4 = pd.read_csv(os.path.join(base_path, file_4), header=None, low_memory=False)

df = pd.concat([df1, df2, df3, df4], ignore_index=True)
df.columns = names
df = df.rename(columns={'Label': 'label'})

""" num_cols = df.select_dtypes(include=['number']).columns.tolist()
print(f"Numerical columns: {len(num_cols)}")
scaler = MinMaxScaler()

df[num_cols] = scaler.fit_transform(df[num_cols]) """

###########
""" # Before scaling
sns.boxplot(data=df[['sbytes', 'ct_dst_src_ltm', 'sttl']])
plt.title("Boxplot without scaling")
plt.show()

# After Min-Max scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[['sbytes', 'ct_dst_src_ltm', 'sttl']])
scaled_df = pd.DataFrame(scaled_features, columns=['sbytes', 'ct_dst_src_ltm', 'sttl'])

sns.boxplot(data=scaled_df)
plt.title("Boxplot with Min-Max scaling")
plt.show()
exit(0) """
###########


LESS_DATA = 10_000

df = df.sample(n=LESS_DATA, random_state=42)

print(f"Loaded: {len(df)} rows")

#df = pd.read_csv(os.path.join(base_path, additional_path, file_training))
df['attack_cat_num'] = pd.factorize(df['attack_cat'])[0] #we encode categorical variable as numerical variable

#compute correlations between numerical features and label category 
label_correlations = df.corr(numeric_only=True)['label'].sort_values(ascending=False)

label_correlations.drop(['id', 'attack_cat_num', 'label'], inplace=True, errors='ignore')

print("Top 10 correlations with label:")
print(label_correlations.head(10))

#most correlated feature: value analysis
most_correlated_feature = label_correlations.index[0]
#most_correlated_feature = label_correlations.index[1]
#most_correlated_feature = 'Stime'
#most_correlated_feature = 'ct_dst_sport_ltm'
#most_correlated_feature = 'rate' #ok
#most_correlated_feature = 'ct_src_dport_ltm'
#most_correlated_feature = 'dloss' #ok
#most_correlated_feature = 'sbytes'


mean_with_0 = df[df['label'] == 0][most_correlated_feature].mean()
mean_with_1 = df[df['label'] == 1][most_correlated_feature].mean()

stddev_with_0 = df[df['label'] == 0][most_correlated_feature].std()
stddev_with_1 = df[df['label'] == 1][most_correlated_feature].std()

#95% confidence interval
n_0 = df[df['label'] == 0][most_correlated_feature].count()
n_1 = df[df['label'] == 1][most_correlated_feature].count()

conf_level = 0.95
alpha = 1 - conf_level

ci_0 = stats.t.interval(
    conf_level, n_0-1, loc=mean_with_0, scale=stddev_with_0/(n_0**0.5)
)
ci_1 = stats.t.interval(
    conf_level, n_1-1, loc=mean_with_1, scale=stddev_with_1/(n_1**0.5)
)

print(f"Average of {most_correlated_feature} when there is NO ATTACK: {mean_with_0} ± {stddev_with_0}")
print(f"95% confidence interval: [{float(ci_0[0])}, {float(ci_0[1])}]")
print(f"Average of {most_correlated_feature} when there is an ATTACK: {mean_with_1} ± {stddev_with_1}")
print(f"95% confidence interval: [{float(ci_1[0])}, {float(ci_1[1])}]\n")

#median
median_with_0 = df[df['label'] == 0][most_correlated_feature].median()
median_with_1 = df[df['label'] == 1][most_correlated_feature].median()

def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
    medians = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = data.sample(n, replace=True)
        medians.append(sample.median())
    
    lower = np.percentile(medians, (1 - ci) / 2 * 100)
    upper = np.percentile(medians, (1 + ci) / 2 * 100)
    
    return lower, upper

""" def bootstrap_ci_2(data, ci=0.95):
    n_bootstrap=1000

    medians = []
    n = len(data)
    for _ in range(n_bootstrap):
        sample = []

        for j in range(n):
            sample.append(random.choice(data))
        
        sample.sort()
        medians.append(0.5*(sample[int(np.floor(n/2))]+sample[int(np.floor(n/2))+1]))
    medians.sort()
    return medians[25], medians[975] """

""" ci_median_0 = bootstrap_ci(df[df['label'] == 0][most_correlated_feature])
ci_median_1 = bootstrap_ci(df[df['label'] == 1][most_correlated_feature])
 """
print(f"Median of {most_correlated_feature} when there is NO ATTACK: {median_with_0}")
#print(f"Bootstrap 95% confidence interval for median: [{ci_median_0[0]}, {ci_median_0[1]}]")
print(f"Median of {most_correlated_feature} when there is an ATTACK: {median_with_1}")
#print(f"Bootstrap 95% confidence interval for median: [{ci_median_1[0]}, {ci_median_1[1]}]")

q25_with_0 = df[df['label'] == 0][most_correlated_feature].quantile(0.25)
q75_with_0 = df[df['label'] == 0][most_correlated_feature].quantile(0.75)
q25_with_1 = df[df['label'] == 1][most_correlated_feature].quantile(0.25)
q75_with_1 = df[df['label'] == 1][most_correlated_feature].quantile(0.75)

print(f"25° percentile (Q1) of {most_correlated_feature} when there is NO ATTACK: {q25_with_0}")
print(f"75° percentile (Q3) of {most_correlated_feature} when there is NO ATTACK: {q75_with_0}")
print(f"25° percentile (Q1) of {most_correlated_feature} when there is an ATTACK: {q25_with_1}")
print(f"75° percentile (Q3) of {most_correlated_feature} when there is an ATTACK: {q75_with_1}")


""" plt.figure(figsize=(8, 6))
sns.boxplot(x='label', y=most_correlated_feature, data=df)
plt.title(f'Boxplot of {most_correlated_feature} per label (0=NO ATTACK, 1=ATTACK)')
plt.xlabel('Label')
plt.ylabel(most_correlated_feature)
plt.show() """

""" sns.kdeplot(data=df, x=most_correlated_feature, hue='label', common_norm=False)
plt.title(f'Distribution of {most_correlated_feature} by Label')
plt.show() """



group_0 = df[df['label'] == 0][most_correlated_feature]
group_1 = df[df['label'] == 1][most_correlated_feature]

t_stat, t_p = ttest_ind(group_0, group_1, equal_var=False)
u_stat, u_p = mannwhitneyu(group_0, group_1, alternative='two-sided')

print(f"\n")
print(f"T-test p-value: {t_p}")
print(f"Mann-Whitney U test p-value: {u_p}")

mean_diff = group_1.mean() - group_0.mean()
median_diff = group_1.median() - group_0.median()
print(f"Mean difference: {mean_diff}")
print(f"Median difference: {median_diff}")

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
    return (x.mean() - y.mean()) / pooled_std

print(f"Cohen's d: {cohens_d(group_1, group_0)}")

""" plt.figure(figsize=(8, 5))
sns.stripplot(x='label', y=most_correlated_feature, data=df, alpha=0.3)
plt.title(f'Correlation between label and {most_correlated_feature}')
plt.xlabel('Label')
plt.ylabel(most_correlated_feature)
plt.show() """

plt.figure(figsize=(8, 6))
sns.boxplot(x='label', y=most_correlated_feature, data=df)
plt.title(f'{most_correlated_feature} by Label (0 = No Attack, 1 = Attack)')
plt.xlabel('Label')
plt.ylabel(most_correlated_feature)
plt.show()
