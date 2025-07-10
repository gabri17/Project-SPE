import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from data_loader import UNSWNB15Loader
import seaborn as sns

class FeatureAnalysis:
    def __init__(self, df, feature, attack_cat_col='attack_cat'):
        self.df = df
        self.feature = feature
        self.label_col = attack_cat_col
        self.groups = {
            cat: df[df[attack_cat_col] == cat][feature]
            for cat in df[attack_cat_col].unique()
        }

    def mean_std(self):
        return {
            cat: (group.mean(), group.std())
            for cat, group in self.groups.items()
        }

    def mean_confidence_interval(self, conf_level=0.95):
        ci = {}
        for cat, group in self.groups.items():
            n = len(group)
            mean = group.mean()
            std = group.std()
            if n > 1:
                interval = stats.t.interval(conf_level, n-1, loc=mean, scale=std/(n**0.5))
            else:
                interval = (mean, mean)
            ci[cat] = interval
        return ci

    def median(self):
        return {cat: group.median() for cat, group in self.groups.items()}
    
    def median_confidence_interval(self, data=None, conf_level=0.95):
        n = len(data)
        alpha = 1 - conf_level
        k = int(np.floor(stats.binom.ppf(alpha / 2, n, 0.5)))
        lower_idx = k
        upper_idx = n - k - 1
        sorted_data = np.sort(data)
        lower = sorted_data[lower_idx] if lower_idx < n else sorted_data[0]
        upper = sorted_data[upper_idx] if upper_idx >= 0 else sorted_data[-1]
        return lower, upper
    
    def median_confidence_intervals(self, conf_level=0.95):
        ci = {}
        for cat, group in self.groups.items():
            ci[cat] = self.median_confidence_interval(group, conf_level)
        return ci

    def quantiles(self):
        return {
            cat: (group.quantile(0.25), group.quantile(0.75))
            for cat, group in self.groups.items()
        }

    def plot_boxplot(self, save_path=None):
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=self.label_col, y=self.feature, data=self.df)
        plt.title(f'Boxplot of {self.feature} per attack_cat (0=NO ATTACK, 1=ATTACK)')
        plt.xlabel('Attack category')
        plt.ylabel(self.feature)
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def write_results(self, file_path):
        means_stds = self.mean_std()
        mean_cis = self.mean_confidence_interval()
        medians = self.median()
        #median_cis = self.median_confidence_intervals()
        quantiles = self.quantiles()

        with open(file_path, "w") as f:
            f.write(f"Feature: {self.feature}\n")
            for cat in self.groups:
                mean, std = means_stds[cat]
                ci = mean_cis[cat]
                median = medians[cat]
                #median_ci = median_cis[cat]
                q25, q75 = quantiles[cat]
                f.write(f"\nGroup {cat} ({len(self.groups[cat])} records):\n")
                f.write(f"  Mean: {mean} +- {std}\n")
                f.write(f"  95% confidence interval (mean): [{float(ci[0])}, {float(ci[1])}]\n")
                f.write(f"  Median: {median}\n")
                #f.write(f"  95% confidence interval (median): [{float(median_ci[0])}, {float(median_ci[1])}]\n")
                f.write(f"  25th percentile (Q1): {q25}\n")
                f.write(f"  75th percentile (Q3): {q75}\n")

    def print_results(self):
        means_stds = self.mean_std()
        mean_cis = self.mean_confidence_interval()
        medians = self.median()
        #median_cis = self.median_confidence_intervals()
        quantiles = self.quantiles()

        print(f"Feature: {self.feature}")
        for cat in self.groups:
            mean, std = means_stds[cat]
            ci = mean_cis[cat]
            median = medians[cat]
            #median_ci = median_cis[cat]
            q25, q75 = quantiles[cat]
            print(f"\nGroup {cat} ({len(self.groups[cat])} records):")
            print(f"  Mean: {mean} +- {std}")
            print(f"  95% confidence interval (mean): [{float(ci[0])}, {float(ci[1])}]")
            print(f"  Median: {median}")
            #print(f"  95% confidence interval (median): [{float(median_ci[0])}, {float(median_ci[1])}]")
            print(f"  25th percentile (Q1): {q25}")
            print(f"  75th percentile (Q3): {q75}")

def main():
    loader = UNSWNB15Loader()
    df = loader.load()
    df['attack_cat'] = df['attack_cat'].fillna('Normal')
    df['attack_cat'] = df['attack_cat'].replace('', 'Normal')

    df['attack_cat'] = df['attack_cat'].str.strip() 
    df['attack_cat'] = df['attack_cat'].replace('Backdoors', 'Backdoor')

    print("Dataset loaded successfully.")

    feature = 'dmeansz'
    analysis = FeatureAnalysis(df, feature)

    analysis.print_results()
    analysis.write_results(f"analysis/correlation_with_attack_cat/feature_analysis_{feature}.txt")
    #analysis.plot_boxplot(f"analysis/feature_analysis_boxplot_{feature}_by_attack_cat.png")

if __name__ == "__main__":
    main()