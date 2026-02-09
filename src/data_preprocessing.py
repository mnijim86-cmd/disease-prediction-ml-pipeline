"""
================================================================================
Data Preprocessing Module
================================================================================
This module handles loading the healthcare dataset, cleaning the data,
and preparing it for machine learning model training.

Key functions:
- load_data: Load the CSV dataset from file
- clean_data: Handle missing values and standardize data
- parse_symptoms: Extract symptoms from comma-separated string
- get_unique_symptoms: Get all unique symptoms in the dataset
- get_data_summary: Generate dataset statistics

Author: Mahmoud Nijim
================================================================================
"""

# ==============================================================================
# IMPORT REQUIRED LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
from typing import Tuple, List, Set


# ==============================================================================
# DATA LOADING FUNCTION
# ==============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the healthcare dataset from a CSV file.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing healthcare data
    
    Returns:
    --------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame
    """
    # Use pandas to read the CSV file
    df = pd.read_csv(filepath)
    return df


# ==============================================================================
# DATA CLEANING FUNCTION
# ==============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and standardizing data.
    
    This function performs the following cleaning steps:
    1. Remove rows with missing disease labels
    2. Fill missing symptoms with empty string
    3. Standardize gender values to title case
    4. Ensure age is numeric and within valid range
    5. Standardize disease names
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw healthcare dataset
    
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset ready for feature engineering
    """
    # Create a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Step 1: Remove rows where Disease (target) is missing
    # We can't train on samples without labels
    df_clean = df_clean.dropna(subset=['Disease'])
    
    # Step 2: Fill missing symptoms with empty string
    # This allows us to process symptoms uniformly
    df_clean['Symptoms'] = df_clean['Symptoms'].fillna('')
    
    # Step 3: Standardize gender values to title case
    # Converts "male", "MALE", "Male" all to "Male"
    df_clean['Gender'] = df_clean['Gender'].str.strip().str.title()
    
    # Step 4: Ensure Age is numeric and clip to valid range
    # Convert non-numeric ages to NaN, then remove those rows
    df_clean['Age'] = pd.to_numeric(df_clean['Age'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Age'])
    
    # Clip age to reasonable range (0-120 years)
    df_clean['Age'] = df_clean['Age'].clip(0, 120)
    
    # Step 5: Standardize disease names to title case
    df_clean['Disease'] = df_clean['Disease'].str.strip().str.title()
    
    # Step 6: Compute Symptom_Count if not present
    # Count the number of symptoms for each patient
    if 'Symptom_Count' not in df_clean.columns:
        df_clean['Symptom_Count'] = df_clean['Symptoms'].apply(
            lambda x: len([s for s in str(x).split(',') if s.strip()]) if pd.notna(x) and x != '' else 0
        )
    
    return df_clean


# ==============================================================================
# SYMPTOM PARSING FUNCTIONS
# ==============================================================================

def parse_symptoms(symptom_string: str) -> List[str]:
    """
    Parse the comma-separated symptom string into a list of individual symptoms.
    
    Example: "fever, headache, cough" -> ["fever", "headache", "cough"]
    
    Parameters:
    -----------
    symptom_string : str
        Comma-separated string of symptoms
    
    Returns:
    --------
    List[str]
        List of individual symptoms, cleaned and lowercase
    """
    # Handle missing or empty symptom strings
    if pd.isna(symptom_string) or symptom_string == '':
        return []
    
    # Split by comma and clean each symptom
    # strip() removes whitespace, lower() converts to lowercase
    symptoms = [s.strip().lower() for s in str(symptom_string).split(',')]
    
    # Remove any empty strings from the list
    symptoms = [s for s in symptoms if s]
    
    return symptoms


def get_unique_symptoms(df: pd.DataFrame) -> List[str]:
    """
    Extract all unique symptoms from the entire dataset.
    
    This function:
    1. Parses each row's symptoms
    2. Adds them to a set (automatically removes duplicates)
    3. Returns sorted list of unique symptoms
    
    Parameters:
    -----------
    df : pd.DataFrame
        Healthcare dataset with 'Symptoms' column
    
    Returns:
    --------
    List[str]
        Sorted list of all unique symptoms in the dataset
    """
    # Use a set to automatically handle duplicates
    all_symptoms: Set[str] = set()
    
    # Loop through each row and extract symptoms
    for symptoms in df['Symptoms']:
        parsed = parse_symptoms(symptoms)
        all_symptoms.update(parsed)  # Add all symptoms to the set
    
    # Return as sorted list for consistent ordering
    return sorted(list(all_symptoms))


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_unique_diseases(df: pd.DataFrame) -> List[str]:
    """
    Get all unique diseases from the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Healthcare dataset with 'Disease' column
    
    Returns:
    --------
    List[str]
        Sorted list of all unique diseases
    """
    return sorted(df['Disease'].unique().tolist())


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive summary of the dataset for exploratory data analysis.
    
    This summary includes:
    - Total samples and features
    - Number of unique diseases and symptoms
    - Age statistics (min, max, mean, median)
    - Gender distribution
    - Disease distribution
    - Symptom count statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Healthcare dataset
    
    Returns:
    --------
    dict
        Dictionary containing all summary statistics
    """
    summary = {
        # Basic counts
        'total_samples': len(df),
        'num_features': len(df.columns),
        'num_diseases': df['Disease'].nunique(),
        'num_unique_symptoms': len(get_unique_symptoms(df)),
        
        # Age statistics
        'age_stats': {
            'min': df['Age'].min(),
            'max': df['Age'].max(),
            'mean': df['Age'].mean(),
            'median': df['Age'].median()
        },
        
        # Categorical distributions
        'gender_distribution': df['Gender'].value_counts().to_dict(),
        'disease_distribution': df['Disease'].value_counts().to_dict(),
        
        # Symptom count statistics
        'symptom_count_stats': {
            'min': df['Symptom_Count'].min(),
            'max': df['Symptom_Count'].max(),
            'mean': df['Symptom_Count'].mean()
        }
    }
    
    return summary
