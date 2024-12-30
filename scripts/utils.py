# Utility functions for performing modular and reusable EDA tasks.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load Dataset
def load_dataset(file_path):
    """Load a dataset from a given file path."""
    return pd.read_csv(file_path)

# Data Summarization
def summarize_data(df):
    """Print a summary of the dataset."""
    print("Dataset Info:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())


# Missing Values Analysis
def analyze_missing_values(df):
    """Check and visualize missing values."""
    print("Missing Values:")
    print(df.isnull().sum())
    msno.matrix(df)
    plt.show()

# Categorical Data Insights
def analyze_categorical_data(df, columns):
    """Print value counts for specified categorical columns."""
    for col in columns:
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print("\n")

# Univariate Analysis
def plot_histogram(df, column, bins=50, color='blue', alpha=0.7):
    """Plot a histogram for a given column."""
    df[column].plot(kind='hist', bins=bins, color=color, alpha=alpha)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_bar_chart(df, column, palette='viridis'):
    """Plot a bar chart for a given column."""
    sns.countplot(data=df, x=column, palette=palette)
    plt.title(f'Count of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Correlation Analysis
def plot_correlation_matrix(df, annot=True, cmap='coolwarm'):
    """Plot a heatmap of the correlation matrix."""
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=annot, cmap=cmap, fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

# Scatter Plot Analysis
def plot_scatter(df, x_col, y_col, hue_col=None, palette='viridis'):
    """Plot a scatterplot for two columns, optionally colored by a third column."""
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette)
    plt.title(f'{y_col} vs. {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if hue_col:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

# Box Plot Analysis
def plot_boxplot(df, x_col, y_col=None, palette='Set2'):
    """Plot a boxplot for a given column or grouped by another column."""
    sns.boxplot(data=df, x=x_col, y=y_col, palette=palette)
    plt.title(f'Boxplot for {x_col}' if not y_col else f'{y_col} by {x_col}')
    plt.xticks(rotation=45)
    plt.show()

# Example of reusable code
def save_plot(file_name):
    """Save the current plot to a file."""
    plt.savefig(file_name, bbox_inches='tight')
