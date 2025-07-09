from data_loader import UNSWNB15Loader

loader = UNSWNB15Loader()
df = loader.load()

print("Dataset loaded successfully.")

#1) compute correlations between numerical features and label
label_correlations = df.corr(numeric_only=True)['label'].drop('label').sort_values(ascending=False)

TO_CONSIDER = 20

print(f"Top {TO_CONSIDER} positive correlations with label:")
print(label_correlations.head(TO_CONSIDER))

print(f"\nTop {TO_CONSIDER} negative correlations with label:")
print(label_correlations.tail(TO_CONSIDER))

with open("analysis/correlation_results.txt", "w") as f:
    f.write(f"\n\nTop {TO_CONSIDER} positive correlations with label:\n")
    f.write(label_correlations.head(TO_CONSIDER).to_string())
    f.write(f"\n\nTop {TO_CONSIDER} negative correlations with label:\n")
    f.write(label_correlations.tail(TO_CONSIDER).to_string())
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    f.write(f"\n\nCategorical columns: {categorical_cols.tolist()}\n")