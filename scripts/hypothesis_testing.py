import pandas as pd 
from scipy.stats import chi2_contingency, ttest_ind, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_path):
    """Load the dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        print("Data successfully loaded!")
        # df = df[['Gender', 'Province', 'PostalCode', 'TotalPremium', 'TotalClaims']]
        print(df.info())  # Display dataset details
        return df
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid file path.")
        return None

def calculate_kpis(df):
    """Add KPI columns to the dataset."""
    df['ClaimRate'] = df['TotalClaims'] / (df['TotalPremium'] + 1e-6)  # Avoid division by zero
    df['ProfitMargin'] = df['TotalPremium'] - df['TotalClaims']
    print("KPIs calculated: ClaimRate and ProfitMargin")
    return df


def segment_data(df, column, group_a_value, group_b_value):
    """Segment the dataset into two groups based on a column and specified values."""
    group_a = df[df[column] == group_a_value]
    group_b = df[df[column] == group_b_value]
    return group_a, group_b

def chi_square_test(df, group_column, target_column):
    """Perform Chi-Square Test for independence."""
    contingency_table = pd.crosstab(df[group_column], df[target_column])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p


def t_test(group_a, group_b, column):
    """Perform T-Test for numerical data."""
    t_stat, p_value = ttest_ind(group_a[column], group_b[column], equal_var=False)
    return t_stat, p_value


def anova_test(groups, column):
    """Perform ANOVA for multiple groups."""
    f_stat, p_value = f_oneway(*[group[column] for group in groups])
    return f_stat, p_value


