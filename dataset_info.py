import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

base_path = os.path.join("datasets", "CSV_Files")
additional_path = os.path.join('training_testing_sets')

file_1 = 'UNSW-NB15_1.csv'
file_2 = 'UNSW-NB15_2.csv'
file_3 = 'UNSW-NB15_3.csv'
file_4 = 'UNSW-NB15_4.csv'
file_training = 'UNSW_NB15_training-set.csv'
file_testing = ''

""" df1 = pd.read_csv(os.path.join(base_path, file_1))
print(len(df1)) #700k

df2 = pd.read_csv(os.path.join(base_path, file_2))
print(len(df2)) #700k

df3 = pd.read_csv(os.path.join(base_path, file_3))
print(len(df3)) #700k

df4 = pd.read_csv(os.path.join(base_path, file_4))
print(len(df4)) #left 440k
 """
df = pd.read_csv(os.path.join(base_path, additional_path, file_training))
df['attack_cat_num'] = pd.factorize(df['attack_cat'])[0]

toCompare = 'attack_cat_num'

# Calcolo correlazione tra feature numeriche e toCompare
correlazioni = df.corr(numeric_only=True)[toCompare].sort_values(ascending=False)

# Mostriamo le prime 10 feature più correlate
print(f"Top 10 correlazioni positive con {toCompare}:")
print(correlazioni.head(10))

print(f"\nTop 10 correlazioni negative con {toCompare}:")
print(correlazioni.tail(10))

interestingFeature = 'sttl' #sttl, ct_dst_sport_ltm, swin

mean_interestingFeature_label_0 = df[df['label'] == 0][interestingFeature].mean()
mean_interestingFeature_label_1 = df[df['label'] == 1][interestingFeature].mean()

stddev_interestingFeature_label_0 = df[df['label'] == 0][interestingFeature].std()
stddev_interestingFeature_label_1 = df[df['label'] == 1][interestingFeature].std()


print(f"Media di {interestingFeature} quando label = 0: {mean_interestingFeature_label_0} +- {stddev_interestingFeature_label_0}")
print(f"Media di {interestingFeature} quando label = 1: {mean_interestingFeature_label_1} +- {stddev_interestingFeature_label_1}")

# Visualizzazione heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title("Heatmap delle correlazioni tra feature numeriche")
plt.show()
