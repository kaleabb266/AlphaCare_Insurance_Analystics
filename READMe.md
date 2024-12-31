# Statistical Modeling and Hypothesis Testing Repository

## Overview

This repository contains Jupyter notebooks and scripts for statistical modeling, hypothesis testing, and exploratory data analysis (EDA). The primary focus is on analyzing insurance data to derive insights and validate hypotheses regarding risk and profitability across different demographics and regions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Notebooks](#notebooks)
  - [Statistical Modeling](#statistical-modeling)
  - [Hypothesis Testing](#hypothesis-testing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Sources](#data-sources)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started, clone the repository and install the required packages. You can use the following commands:

```
git clone https://github.com/yourusername/repo-name.git
cd repo-name
pip install -r requirements.txt
```

Make sure you have Jupyter Notebook installed. If not, you can install it using:

```
pip install notebook
```

## Usage

To run the notebooks, navigate to the `notebooks` directory and start Jupyter Notebook:

```
jupyter notebook
```

Open the desired notebook and execute the cells to see the analysis in action.

## Notebooks

### Statistical Modeling

- **File:** `notebooks/Statistical_Modeling.ipynb`
- **Description:** This notebook focuses on building statistical models to predict insurance claims and assess the impact of various factors on risk.

### Hypothesis Testing

- **File:** `notebooks/hypothesis_testing.ipynb`
- **Description:** This notebook conducts hypothesis tests to evaluate risk differences across provinces, zip codes, and genders using ANOVA and T-tests.

### Exploratory Data Analysis

- **File:** `notebooks/EDA.ipynb`
- **Description:** This notebook performs exploratory data analysis to visualize and summarize the dataset, identifying trends and patterns in the data.

## Data Sources

The data used in this repository is sourced from:

- `../data/MachineLearningRating_v3.csv`: Contains insurance-related data for analysis.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Pandas](https://pandas.pydata.org/) for data manipulation.
- [Seaborn](https://seaborn.pydata.org/) for data visualization.
- [LIME](https://github.com/marcotcr/lime) for model interpretability.