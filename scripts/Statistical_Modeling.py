import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap

# Define preprocessing functions
def load_data(file_path):
    """
    Load the dataset from a CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Perform data preprocessing including handling missing values, encoding, and feature scaling.
    """
    # Select relevant columns based on insurance analysis
    selected_columns = [
        'TransactionMonth', 'IsVATRegistered', 'MaritalStatus', 'Gender',
        'Province', 'VehicleType', 'RegistrationYear', 'SumInsured',
        'CalculatedPremiumPerTerm', 'TotalPremium', 'TotalClaims'
    ]
    df = df[selected_columns]

    # Handle missing values
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Impute missing values
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_num = SimpleImputer(strategy='mean')

    df.loc[:, categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
    df.loc[:, numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

    # Encode categorical variables
    encoder = OneHotEncoder(sparse_output=False, drop='first') 
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))

    # Scale numerical variables
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_features, columns=numerical_cols)

    # Combine the processed numerical and categorical data
    processed_df = pd.concat([scaled_df, encoded_df], axis=1)

    return processed_df

def split_data(df, target_column):
    """
    Split the data into training and testing sets.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def visualize_data(df):
    """
    Generate visualizations for exploratory data analysis.
    """
    # Visualize distribution of target variables
    plt.figure(figsize=(8, 6))
    sns.histplot(df['TotalPremium'], kde=True, bins=30)
    plt.title("Distribution of TotalPremium")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(df['TotalClaims'], kde=True, bins=30)
    plt.title("Distribution of TotalClaims")
    plt.show()

    # Correlation heatmap for selected features
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple models.
    """
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }

    results = {}
    feature_importances = {}


    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        results[model_name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

        print(f"{model_name} Results: MSE={mse}, MAE={mae}, R2={r2}\n")

        # Save feature importances for tree-based models
        if hasattr(model, 'feature_importances_'):
            feature_importances[model_name] = model.feature_importances_

    return results, feature_importances, models


def interpret_model_with_shap(model, X_train):
    """
    Use SHAP to interpret the model's predictions.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train)

    # Summary plot of SHAP values
    shap.summary_plot(shap_values, X_train)

    # Bar plot of mean absolute SHAP values
    shap.summary_plot(shap_values, X_train, plot_type="bar")

# The main logic has been removed to make this script importable in a Jupyter Notebook.
