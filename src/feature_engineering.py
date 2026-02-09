"""
================================================================================
Feature Engineering Module
================================================================================
This module transforms raw data into features suitable for machine learning.

Key techniques used:
- Binary encoding of symptoms (Multi-Label Binarization)
- One-hot encoding for categorical variables (Gender)
- Age binning and normalization
- Feature combination

The main output is a feature matrix X and target vector y for ML training.

Author: Mahmoud Nijim
================================================================================
"""

# ==============================================================================
# IMPORT REQUIRED LIBRARIES
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from typing import Tuple, List, Dict

# Import our symptom parsing functions from data preprocessing module
from src.data_preprocessing import parse_symptoms, get_unique_symptoms


# ==============================================================================
# SYMPTOM FEATURE ENGINEERING
# ==============================================================================

def create_symptom_features(df: pd.DataFrame, symptom_list: List[str] = None) -> pd.DataFrame:
    """
    Create binary features for each symptom (Multi-Label Binarization).
    
    This converts symptoms from text to numeric format:
    - Each unique symptom becomes a column
    - Value is 1 if patient has that symptom, 0 otherwise
    
    Example:
        Original: "fever, headache"
        Result: symptom_fever=1, symptom_headache=1, symptom_cough=0, ...
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'Symptoms' column
    symptom_list : List[str], optional
        List of symptoms to create features for. If None, uses all symptoms in data.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with binary symptom features (0 or 1 for each symptom)
    """
    # Get symptom list if not provided
    if symptom_list is None:
        symptom_list = get_unique_symptoms(df)
    
    # Initialize symptom feature matrix with all zeros
    # Column names are prefixed with 'symptom_' for clarity
    symptom_features = pd.DataFrame(
        0, 
        index=df.index, 
        columns=[f'symptom_{s.replace(" ", "_")}' for s in symptom_list]
    )
    
    # Loop through each row and set 1s where symptoms are present
    for idx, symptoms in df['Symptoms'].items():
        parsed = parse_symptoms(symptoms)
        for symptom in parsed:
            col_name = f'symptom_{symptom.replace(" ", "_")}'
            if col_name in symptom_features.columns:
                symptom_features.loc[idx, col_name] = 1
    
    return symptom_features


# ==============================================================================
# DEMOGRAPHIC FEATURE ENGINEERING
# ==============================================================================

def create_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from demographic data (Age and Gender).
    
    This function applies several transformations:
    1. Age normalization (divide by 100 to scale to 0-1.2 range)
    2. Age binning into meaningful groups (child, teen, adult, etc.)
    3. One-hot encoding of age groups
    4. One-hot encoding of gender
    5. Symptom count as a feature
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with 'Age' and 'Gender' columns
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with demographic features
    """
    demo_features = pd.DataFrame(index=df.index)
    
    # -------------------------------------------------------------------------
    # Age Features
    # -------------------------------------------------------------------------
    
    # Normalized age (scaled to approximately 0-1 range)
    demo_features['age_normalized'] = df['Age'] / 100.0
    
    # Age binning - create meaningful age groups
    # This captures non-linear relationships between age and disease
    age_bins = [0, 12, 18, 35, 50, 65, 120]
    age_labels = ['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'senior']
    demo_features['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
    
    # One-hot encode age groups (convert categorical to binary columns)
    age_dummies = pd.get_dummies(demo_features['age_group'], prefix='age')
    demo_features = pd.concat([demo_features, age_dummies], axis=1)
    demo_features = demo_features.drop('age_group', axis=1)  # Remove original column
    
    # -------------------------------------------------------------------------
    # Gender Features
    # -------------------------------------------------------------------------
    
    # One-hot encode gender (Male, Female, Other become separate binary columns)
    gender_dummies = pd.get_dummies(df['Gender'], prefix='gender')
    demo_features = pd.concat([demo_features, gender_dummies], axis=1)
    
    # -------------------------------------------------------------------------
    # Other Features
    # -------------------------------------------------------------------------
    
    # Symptom count - number of symptoms patient reports
    demo_features['symptom_count'] = df['Symptom_Count']
    
    return demo_features


# ==============================================================================
# MAIN FEATURE PREPARATION FUNCTION
# ==============================================================================

def prepare_features(df: pd.DataFrame, symptom_list: List[str] = None) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, List[str]]:
    """
    Prepare the complete feature matrix and target vector for ML training.
    
    This is the main function that combines all feature engineering steps:
    1. Create symptom features (binary encoding)
    2. Create demographic features (age, gender)
    3. Combine all features into single matrix X
    4. Encode target variable (disease labels) into numbers
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned healthcare dataset
    symptom_list : List[str], optional
        Predefined list of symptoms for consistent feature creation
    
    Returns:
    --------
    Tuple containing:
        - X: Feature matrix (pd.DataFrame) with all features
        - y: Encoded target vector (pd.Series) with disease labels as numbers
        - label_encoder: Fitted LabelEncoder to convert between disease names and numbers
        - symptom_list: List of symptoms used for features
    """
    # Get symptom list if not provided
    if symptom_list is None:
        symptom_list = get_unique_symptoms(df)
    
    # Create symptom features using multi-label binarization
    symptom_features = create_symptom_features(df, symptom_list)
    
    # Create demographic features (age and gender)
    demo_features = create_demographic_features(df)
    
    # Combine all features into one feature matrix
    X = pd.concat([demo_features, symptom_features], axis=1)
    
    # Encode target variable (Disease) from strings to integers
    # LabelEncoder maps each disease to a unique number
    label_encoder = LabelEncoder()
    y = pd.Series(label_encoder.fit_transform(df['Disease']), index=df.index)
    
    return X, y, label_encoder, symptom_list


# ==============================================================================
# SINGLE PREDICTION PREPARATION
# ==============================================================================

def prepare_single_prediction(
    age: int, 
    gender: str, 
    symptoms: List[str], 
    symptom_list: List[str],
    feature_columns: List[str]
) -> pd.DataFrame:
    """
    Prepare features for a single prediction from user input.
    
    This function is used during interactive prediction to convert
    user input into the same format as training data.
    
    Parameters:
    -----------
    age : int
        Patient's age
    gender : str
        Patient's gender (Male/Female/Other)
    symptoms : List[str]
        List of patient's symptoms
    symptom_list : List[str]
        Complete list of all possible symptoms (same as training)
    feature_columns : List[str]
        List of feature column names (same as training)
    
    Returns:
    --------
    pd.DataFrame
        Feature vector ready for model prediction
    """
    # Create a single-row DataFrame with patient data
    data = {
        'Age': [age],
        'Gender': [gender.title()],
        'Symptoms': [', '.join(symptoms)],
        'Symptom_Count': [len(symptoms)]
    }
    df = pd.DataFrame(data)
    
    # Create symptom features using same process as training
    symptom_features = create_symptom_features(df, symptom_list)
    
    # Create demographic features using same process as training
    demo_features = create_demographic_features(df)
    
    # Combine features
    X = pd.concat([demo_features, symptom_features], axis=1)
    
    # Ensure all columns from training data exist
    # Add missing columns with value 0
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    
    # Reorder columns to match training data exactly
    X = X[feature_columns]
    
    return X


# ==============================================================================
# UTILITY FUNCTION FOR FEATURE NAMES
# ==============================================================================

def get_feature_importance_names(feature_columns: List[str]) -> Dict[str, str]:
    """
    Create human-readable names for feature columns.
    
    Converts technical column names to readable format:
    - "symptom_high_fever" -> "High Fever"
    - "age_senior" -> "Age Senior"
    
    Parameters:
    -----------
    feature_columns : List[str]
        List of feature column names
    
    Returns:
    --------
    Dict[str, str]
        Mapping from column name to readable name
    """
    readable_names = {}
    for col in feature_columns:
        if col.startswith('symptom_'):
            readable_names[col] = col.replace('symptom_', '').replace('_', ' ').title()
        elif col.startswith('age_'):
            readable_names[col] = col.replace('_', ' ').title()
        elif col.startswith('gender_'):
            readable_names[col] = col.replace('_', ' ').title()
        else:
            readable_names[col] = col.replace('_', ' ').title()
    
    return readable_names
