#fammi dei pandas dataframe con dati random
import pandas as pd
import numpy as np

# Crea un DataFrame con dati casuali
df_random = pd.read_csv("r.csv")

#ora crea un altro pandas dataframe con le stesse colonne, ma senza specificare l'header
df_random_no_header = pd.read_csv("r1.csv", header=None)
df_random_no_header.columns = df_random.columns 

print(df_random)
print(df_random_no_header)

print(pd.concat([df_random, df_random_no_header], ignore_index=True, axis=0))