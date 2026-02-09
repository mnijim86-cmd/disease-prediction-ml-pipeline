"""
================================================================================
Sklearn Pipeline Module
================================================================================
This module contains custom sklearn transformers and pipeline utilities for
the disease prediction system.

Custom Transformers:
- SymptomBinarizer: Converts comma-separated symptoms to binary features
- DemographicTransformer: Creates age and gender features

Using Pipeline ensures no data leakage during cross-validation.

Author: Mahmoud Nijim
================================================================================
"""

# ==============================================================================
# IMPORT REQUIRED LIBRARIES
# ==============================================================================
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


# ==============================================================================
# CUSTOM TRANSFORMER: SYMPTOM BINARIZER
# ==============================================================================

class SymptomBinarizer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer that converts comma-separated symptom strings
    into binary feature vectors (Multi-Label Binarization).
    
    During fit():
        - Learns all unique symptoms from the training data
    
    During transform():
        - Creates binary features for each known symptom
        - Unknown symptoms are ignored (no new columns)
    
    This ensures no data leakage during cross-validation.
    """
    
    def __init__(self, symptom_column: str = 'Symptoms'):
        """
        Initialize the SymptomBinarizer.
        
        Parameters:
        -----------
        symptom_column : str
            Name of the column containing comma-separated symptoms
        """
        self.symptom_column = symptom_column
        self.symptom_list_: Optional[List[str]] = None
    
    def _parse_symptoms(self, symptom_string: str) -> List[str]:
        """Parse comma-separated symptom string into list."""
        if pd.isna(symptom_string) or symptom_string == '':
            return []
        symptoms = [s.strip().lower() for s in str(symptom_string).split(',')]
        return [s for s in symptoms if s]
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn all unique symptoms from training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data with 'Symptoms' column
        y : ignored
        
        Returns:
        --------
        self
        """
        all_symptoms = set()
        for symptom_str in X[self.symptom_column]:
            parsed = self._parse_symptoms(symptom_str)
            all_symptoms.update(parsed)
        
        # Store sorted list for consistent column ordering
        self.symptom_list_ = sorted(list(all_symptoms))
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform symptom strings into binary feature matrix.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data with 'Symptoms' column
        
        Returns:
        --------
        np.ndarray
            Binary feature matrix (n_samples, n_symptoms)
        """
        if self.symptom_list_ is None:
            raise ValueError("SymptomBinarizer has not been fitted yet.")
        
        # Create mapping from symptom to column index
        symptom_to_idx = {s: i for i, s in enumerate(self.symptom_list_)}
        n_samples = len(X)
        n_features = len(self.symptom_list_)
        
        # Initialize binary matrix with zeros
        result = np.zeros((n_samples, n_features), dtype=np.float64)
        
        # Fill in 1s where symptoms are present
        for i, symptom_str in enumerate(X[self.symptom_column].values):
            parsed = self._parse_symptoms(symptom_str)
            for symptom in parsed:
                if symptom in symptom_to_idx:
                    result[i, symptom_to_idx[symptom]] = 1.0
        
        return result
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return feature names for the output."""
        if self.symptom_list_ is None:
            return np.array([])
        return np.array([f'symptom_{s.replace(" ", "_")}' for s in self.symptom_list_])


# ==============================================================================
# CUSTOM TRANSFORMER: DEMOGRAPHIC FEATURES
# ==============================================================================

class DemographicTransformer(BaseEstimator, TransformerMixin):
    """
    Custom sklearn transformer for demographic features (Age, Gender).
    
    Creates features:
    - age_normalized: Age divided by 100
    - symptom_count: Number of symptoms
    - age_child, age_teen, etc.: One-hot encoded age groups
    - gender_Male, gender_Female, gender_Other: One-hot encoded gender
    
    During fit():
        - Learns all unique gender values and age bin boundaries
    
    During transform():
        - Creates normalized and one-hot encoded features
    """
    
    def __init__(self, age_column: str = 'Age', gender_column: str = 'Gender',
                 symptom_count_column: str = 'Symptom_Count'):
        """
        Initialize the DemographicTransformer.
        
        Parameters:
        -----------
        age_column : str
            Name of the age column
        gender_column : str
            Name of the gender column
        symptom_count_column : str
            Name of the symptom count column
        """
        self.age_column = age_column
        self.gender_column = gender_column
        self.symptom_count_column = symptom_count_column
        
        # Fixed age bins (not learned from data)
        self.age_bins = [0, 12, 18, 35, 50, 65, 120]
        self.age_labels = ['child', 'teen', 'young_adult', 'adult', 'middle_aged', 'senior']
        
        # Learned during fit
        self.gender_classes_: Optional[List[str]] = None
        self.feature_names_: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Learn gender categories from training data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data with Age, Gender, Symptom_Count columns
        y : ignored
        
        Returns:
        --------
        self
        """
        # Learn unique gender values
        self.gender_classes_ = sorted(X[self.gender_column].unique().tolist())
        
        # Build feature names list
        self.feature_names_ = ['age_normalized', 'symptom_count']
        self.feature_names_.extend([f'age_{label}' for label in self.age_labels])
        self.feature_names_.extend([f'gender_{g}' for g in self.gender_classes_])
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform demographic data into feature matrix.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data with Age, Gender, Symptom_Count columns
        
        Returns:
        --------
        np.ndarray
            Feature matrix (n_samples, n_demographic_features)
        """
        if self.gender_classes_ is None:
            raise ValueError("DemographicTransformer has not been fitted yet.")
        
        n_samples = len(X)
        n_features = len(self.feature_names_)
        result = np.zeros((n_samples, n_features), dtype=np.float64)
        
        # Feature indices
        age_norm_idx = 0
        symptom_count_idx = 1
        age_group_start = 2
        gender_start = age_group_start + len(self.age_labels)
        
        for i in range(n_samples):
            row = X.iloc[i]
            
            # Age normalized
            result[i, age_norm_idx] = row[self.age_column] / 100.0
            
            # Symptom count
            if self.symptom_count_column in X.columns:
                result[i, symptom_count_idx] = row[self.symptom_count_column]
            
            # Age group (one-hot)
            age = row[self.age_column]
            for j, (low, high) in enumerate(zip(self.age_bins[:-1], self.age_bins[1:])):
                if low <= age < high:
                    result[i, age_group_start + j] = 1.0
                    break
            
            # Gender (one-hot)
            gender = row[self.gender_column]
            if gender in self.gender_classes_:
                gender_idx = self.gender_classes_.index(gender)
                result[i, gender_start + gender_idx] = 1.0
        
        return result
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return feature names for the output."""
        if self.feature_names_ is None:
            return np.array([])
        return np.array(self.feature_names_)


# ==============================================================================
# COMBINED FEATURE TRANSFORMER
# ==============================================================================

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Combined transformer that applies both SymptomBinarizer and DemographicTransformer.
    
    This is the main transformer used in the disease prediction pipeline.
    It combines symptoms and demographics into a single feature matrix.
    """
    
    def __init__(self):
        """Initialize the FeatureTransformer with sub-transformers."""
        self.symptom_binarizer = SymptomBinarizer()
        self.demographic_transformer = DemographicTransformer()
        self.feature_names_: Optional[List[str]] = None
    
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit both sub-transformers.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Training data with Symptoms, Age, Gender, Symptom_Count columns
        y : ignored
        
        Returns:
        --------
        self
        """
        self.symptom_binarizer.fit(X, y)
        self.demographic_transformer.fit(X, y)
        
        # Combine feature names
        demo_names = list(self.demographic_transformer.get_feature_names_out())
        symptom_names = list(self.symptom_binarizer.get_feature_names_out())
        self.feature_names_ = demo_names + symptom_names
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using both sub-transformers.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Data with Symptoms, Age, Gender, Symptom_Count columns
        
        Returns:
        --------
        np.ndarray
            Combined feature matrix
        """
        symptom_features = self.symptom_binarizer.transform(X)
        demographic_features = self.demographic_transformer.transform(X)
        
        # Concatenate: demographics first, then symptoms
        return np.hstack([demographic_features, symptom_features])
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return combined feature names."""
        if self.feature_names_ is None:
            return np.array([])
        return np.array(self.feature_names_)
    
    @property
    def symptom_list_(self) -> List[str]:
        """Get the learned symptom list."""
        return self.symptom_binarizer.symptom_list_ or []


# ==============================================================================
# PIPELINE CREATION FUNCTIONS
# ==============================================================================

def create_disease_pipeline(model: Any, scale_features: bool = False) -> Pipeline:
    """
    Create a complete sklearn Pipeline for disease prediction.
    
    The pipeline includes:
    1. FeatureTransformer (symptom binarization + demographics)
    2. Optional StandardScaler (for Logistic Regression)
    3. The ML model (classifier)
    
    Parameters:
    -----------
    model : Any
        Scikit-learn classifier (e.g., RandomForestClassifier)
    scale_features : bool
        Whether to apply StandardScaler (recommended for Logistic Regression)
    
    Returns:
    --------
    Pipeline
        Complete sklearn Pipeline for training and prediction
    """
    steps = [
        ('features', FeatureTransformer())
    ]
    
    if scale_features:
        steps.append(('scaler', StandardScaler()))
    
    steps.append(('classifier', model))
    
    return Pipeline(steps)


def get_feature_names_from_pipeline(pipeline: Pipeline) -> List[str]:
    """
    Extract feature names from a fitted pipeline.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Fitted sklearn Pipeline
    
    Returns:
    --------
    List[str]
        Feature names from the FeatureTransformer step
    """
    if 'features' in pipeline.named_steps:
        feature_transformer = pipeline.named_steps['features']
        return list(feature_transformer.get_feature_names_out())
    return []


def get_symptom_list_from_pipeline(pipeline: Pipeline) -> List[str]:
    """
    Extract the symptom list from a fitted pipeline.
    
    Parameters:
    -----------
    pipeline : Pipeline
        Fitted sklearn Pipeline
    
    Returns:
    --------
    List[str]
        Symptom list learned during training
    """
    if 'features' in pipeline.named_steps:
        feature_transformer = pipeline.named_steps['features']
        return feature_transformer.symptom_list_
    return []
