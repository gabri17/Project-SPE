import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway
from data_loader import UNSWNB15Loader
import seaborn as sns

class FeatureAnalysis:
    def __init__(self, df, feature, measure_unit, label_col='label'):
        self.df = df
        self.feature = feature
        self.measure_unit = measure_unit
        self.label_col = label_col
        self.group_0 = df[df[label_col] == 0][feature]
        self.group_1 = df[df[label_col] == 1][feature]

    def mean_std(self):
        mean_0 = self.group_0.mean()
        mean_1 = self.group_1.mean()
        std_0 = self.group_0.std()
        std_1 = self.group_1.std()
        return mean_0, std_0, mean_1, std_1

    def mean_confidence_interval(self, conf_level=0.95):
        n_0 = len(self.group_0)
        n_1 = len(self.group_1)
        mean_0, std_0, mean_1, std_1 = self.mean_std()
        ci_0 = stats.t.interval(conf_level, n_0-1, loc=mean_0, scale=std_0/(n_0**0.5))
        ci_1 = stats.t.interval(conf_level, n_1-1, loc=mean_1, scale=std_1/(n_1**0.5))
        return ci_0, ci_1

    def median(self):
        return self.group_0.median(), self.group_1.median()
    
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

    def quantiles(self):
        q25_0 = self.group_0.quantile(0.25)
        q75_0 = self.group_0.quantile(0.75)
        q25_1 = self.group_1.quantile(0.25)
        q75_1 = self.group_1.quantile(0.75)
        return q25_0, q75_0, q25_1, q75_1

    def plot_boxplot(self, save_path=None):
        plt.figure(figsize=(5, 6))
        myPalette = {'0': "#1f77b4", '1': "#d62728"}
        ax = sns.boxplot(x=self.label_col, y=self.feature, data=self.df, width=0.3, showfliers=False, palette=myPalette)
        ax.set_xticklabels(['NO ATTACK', 'ATTACK'])
        plt.title(f'Boxplot of {self.feature} per label')
        plt.xlabel('Label')
        plt.ylabel(f"{self.feature} [{self.measure_unit}]")
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def ttest(self):
        t_stat, t_p = ttest_ind(self.group_0, self.group_1, equal_var=False)
        return t_stat, t_p

    def mannwhitney(self):
        u_stat, u_p = mannwhitneyu(self.group_0, self.group_1, alternative='two-sided')
        return u_stat, u_p

    def anova(self):
        stat, p = f_oneway(self.group_0, self.group_1)
        return stat, p

    def mean_median_diff(self):
        mean_diff = self.group_1.mean() - self.group_0.mean()
        median_diff = self.group_1.median() - self.group_0.median()
        return mean_diff, median_diff

    def cohens_d(self):
        x, y = self.group_1, self.group_0
        nx, ny = len(x), len(y)
        pooled_std = np.sqrt(((nx-1)*x.std()**2 + (ny-1)*y.std()**2) / (nx+ny-2))
        return (x.mean() - y.mean()) / pooled_std
    
    def coefficient_of_variation(self):
        cv_0 = self.group_0.std() / self.group_0.mean() if self.group_0.mean() != 0 else np.nan
        cv_1 = self.group_1.std() / self.group_1.mean() if self.group_1.mean() != 0 else np.nan
        return cv_0, cv_1

    def mad_gap(self, data):
        m = np.mean(data)
        mad = np.mean(np.abs(data - m))
        gap = mad / (2 * m) if m != 0 else np.nan
        return mad, gap


    def write_results(self, file_path):
        mean_0, std_0, mean_1, std_1 = self.mean_std()
        ci_0, ci_1 = self.mean_confidence_interval()
        median_ci_0 = self.median_confidence_interval(self.group_0)
        median_ci_1 = self.median_confidence_interval(self.group_1)
        mad_0, gap_0 = self.mad_gap(self.group_0)
        mad_1, gap_1 = self.mad_gap(self.group_1)
        median_0, median_1 = self.median()
        q25_0, q75_0, q25_1, q75_1 = self.quantiles()
        t_stat, t_p = self.ttest()
        u_stat, u_p = self.mannwhitney()
        stat, p = self.anova()
        mean_diff, median_diff = self.mean_median_diff()
        cohens_d = self.cohens_d()
        cv_0, cv_1 = self.coefficient_of_variation()

        with open(file_path, "w") as f:
            f.write(f"Feature: {self.feature}\n")
            f.write(f"Mean when there is NO ATTACK: {mean_0} +- {std_0}\n")
            f.write(f"95% confidence interval: [{float(ci_0[0])}, {float(ci_0[1])}]\n")
            f.write(f"Mean when there is an ATTACK: {mean_1} +- {std_1}\n")
            f.write(f"95% confidence interval: [{float(ci_1[0])}, {float(ci_1[1])}]\n")
            f.write(f"\nMedian of {self.feature} when there is NO ATTACK: {median_0}\n")
            f.write(f"95% confidence interval: [{float(median_ci_0[0])}, {float(median_ci_0[1])}]\n")
            f.write(f"Median of {self.feature} when there is an ATTACK: {median_1}\n")
            f.write(f"95% confidence interval: [{float(median_ci_1[0])}, {float(median_ci_1[1])}]\n")
            f.write(f"\nMin value with NO ATTACK: {self.group_0.min()}\n")
            f.write(f"Max value with NO ATTACK: {self.group_0.max()}\n")
            f.write(f"\nMin value with ATTACK: {self.group_1.min()}\n")
            f.write(f"Max value with ATTACK: {self.group_1.max()}\n")
            f.write(f"\n25th percentile (Q1) when NO ATTACK: {q25_0}\n")
            f.write(f"75th percentile (Q3) when NO ATTACK: {q75_0}\n")
            f.write(f"25th percentile (Q1) when ATTACK: {q25_1}\n")
            f.write(f"75th percentile (Q3) when ATTACK: {q75_1}\n")
            f.write(f"\nCoefficient of Variation when NO ATTACK: {cv_0:.4f}\n")
            f.write(f"Coefficient of Variation when ATTACK: {cv_1:.4f}\n")
            f.write(f"\nMAD when NO ATTACK: {mad_0:.4f}, Gap: {gap_0:.4f}\n")
            f.write(f"MAD when ATTACK: {mad_1:.4f}, Gap: {gap_1:.4f}\n")
            f.write(f"\nT-test p-value: {t_p}\n")
            f.write(f"Mann-Whitney U test p-value: {u_p}\n")
            f.write(f"Anova F-Test p-value: {p:.4g}\n")
            f.write(f"Mean difference: {mean_diff}\n")
            f.write(f"Median difference: {median_diff}\n")
            f.write(f"Cohen's d: {cohens_d}\n")

    def print_results(self):
        mean_0, std_0, mean_1, std_1 = self.mean_std()
        ci_0, ci_1 = self.mean_confidence_interval()
        median_0, median_1 = self.median()
        median_ci_0 = self.median_confidence_interval(self.group_0)
        median_ci_1 = self.median_confidence_interval(self.group_1)
        q25_0, q75_0, q25_1, q75_1 = self.quantiles()
        t_stat, t_p = self.ttest()
        u_stat, u_p = self.mannwhitney()
        stat, p = self.anova()
        mean_diff, median_diff = self.mean_median_diff()
        cohens_d = self.cohens_d()
        cv_0, cv_1 = self.coefficient_of_variation()
        mad_0, gap_0 = self.mad_gap(self.group_0)
        mad_1, gap_1 = self.mad_gap(self.group_1)

        print(f"Average of {self.feature} when there is NO ATTACK: {mean_0} ± {std_0}")
        print(f"95% confidence interval: [{float(ci_0[0])}, {float(ci_0[1])}]")
        print(f"Average of {self.feature} when there is an ATTACK: {mean_1} ± {std_1}")
        print(f"95% confidence interval: [{float(ci_1[0])}, {float(ci_1[1])}]\n")
        print(f"Median of {self.feature} when there is NO ATTACK: {median_0}")
        print(f"95% confidence interval: [{float(median_ci_0[0])}, {float(median_ci_0[1])}]\n")
        print(f"Median of {self.feature} when there is an ATTACK: {median_1}")
        print(f"95% confidence interval: [{float(median_ci_1[0])}, {float(median_ci_1[1])}]\n")
        print(f"25th percentile (Q1) when NO ATTACK: {q25_0}")
        print(f"75th percentile (Q3) when NO ATTACK: {q75_0}")
        print(f"25th percentile (Q1) when ATTACK: {q25_1}")
        print(f"75th percentile (Q3) when ATTACK: {q75_1}")
        print(f"\nMin value with NO ATTACK: {self.group_0.min()}\n")
        print(f"Max value with NO ATTACK: {self.group_0.max()}\n")
        print(f"\nMin value with ATTACK: {self.group_1.min()}\n")
        print(f"Max value with ATTACK: {self.group_1.max()}\n")
        print(f"\nCoefficient of Variation when NO ATTACK: {cv_0:.4f}")
        print(f"Coefficient of Variation when ATTACK: {cv_1:.4f}\n")
        print(f"MAD when NO ATTACK: {mad_0:.4f}, Gap: {gap_0:.4f}")
        print(f"MAD when ATTACK: {mad_1:.4f}, Gap: {gap_1:.4f}\n")
        print(f"\nT-test p-value: {t_p}")
        print(f"Mann-Whitney U test p-value: {u_p}")
        print(f"Anova F-Test p-value={p:.4g}")
        print(f"Mean difference: {mean_diff}")
        print(f"Median difference: {median_diff}")
        print(f"Cohen's d: {cohens_d}")

def main():
    loader = UNSWNB15Loader()
    df = loader.load()
    print("Dataset loaded successfully.")

    """ df['sbytes_over_dbytes'] = df['sbytes'] / df['dbytes']
    df['sbytes_over_dbytes'].replace([np.inf, -np.inf], 0, inplace=True) """

    for (feature, unit) in [('sttl', 'TTL value'), ('ct_state_ttl', 'No. connections'), ('ct_dst_src_ltm', 'No. connections'), ('ct_dst_sport_ltm', 'No. connections'), ('ct_src_dport_ltm', 'No. connections'), ('ct_srv_dst', 'No. connections'), ('ct_srv_src', 'No. connections'), ('ct_src_ ltm', 'No. connections'), ('ct_dst_ltm', 'No. connections'), ('Sload', 'Source load (bits per second)')]:
        analysis = FeatureAnalysis(df, feature, measure_unit=unit)

        analysis.print_results()
        analysis.write_results(f"analysis/correlation_with_label/feature_analysis_{feature}.txt")
        analysis.plot_boxplot(f"analysis/correlation_with_label/feature_analysis_boxplot_{feature}_by_label.png")

if __name__ == "__main__":
    main()