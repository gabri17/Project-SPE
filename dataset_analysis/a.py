import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

#heatmap of correlations among different features
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', center=0)
plt.title("Heatmap delle correlazioni tra feature numeriche")
#plt.show()

#compute correlations between numerical features and attack category 
attack_correlations = df.corr(numeric_only=True)['attack_cat_num'].sort_values(ascending=False)

print(f"Top 10 positive correlations with attack_cat_num:")
print(attack_correlations.head(10))

print(f"\nTop 10 negative correlations with attack_cat_num:")
print(attack_correlations.tail(10))

#compute correlations between numerical features and attack category 
label_correlations = df.corr(numeric_only=True)['label'].sort_values(ascending=False)

print(f"Top 10 positive correlations with label:")
print(label_correlations.head(10))

print(f"\nTop 10 negative correlations with label:")
print(label_correlations.tail(10))