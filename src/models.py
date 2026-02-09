"""
================================================================================
Machine Learning Models Module
================================================================================
This module contains functions for training, evaluating, and comparing
machine learning algorithms for disease prediction.

Algorithms implemented:
1. Logistic Regression - Linear baseline with interpretable coefficients
2. Random Forest - Ensemble of decision trees with feature importance
3. Gradient Boosting - Sequential tree building for strong performance

Each model is evaluated using cross-validation and multiple metrics.

Author: Mahmoud Nijim
================================================================================
"""

# ==============================================================================
# IMPORT REQUIRED LIBRARIES
# ==============================================================================
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn warnings for cleaner output

# Scikit-learn model imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Scikit-learn utilities
from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV,
    StratifiedKFold
)

# Scikit-learn evaluation metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


# ==============================================================================
# MODEL DEFINITIONS
# ==============================================================================

def get_models() -> Dict[str, Any]:
    """
    Return a dictionary of machine learning models with default parameters.
    
    Models:
    -------
    1. Logistic Regression:
       - Linear model that predicts probability of each class
       - Interpretable coefficients show feature importance
       - Good baseline model
    
    2. Random Forest:
       - Ensemble of 100 decision trees
       - Each tree votes for a class, majority wins
       - Captures non-linear relationships
       - Built-in feature importance
    
    3. Gradient Boosting:
       - Sequential tree building
       - Each tree corrects errors of previous trees
       - Often achieves best accuracy
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary mapping model names to model instances
    """
    models = {
        # Logistic Regression - Linear baseline model
        'Logistic Regression': LogisticRegression(
            max_iter=1000,          # Maximum iterations for convergence
            solver='lbfgs',         # Optimization algorithm
            random_state=42         # For reproducibility
        ),
        
        # Random Forest - Ensemble of decision trees
        'Random Forest': RandomForestClassifier(
            n_estimators=100,       # Number of trees in the forest
            max_depth=15,           # Maximum depth of each tree
            min_samples_split=5,    # Minimum samples to split a node
            min_samples_leaf=2,     # Minimum samples at leaf node
            random_state=42,        # For reproducibility
            n_jobs=-1               # Use all CPU cores for faster training
        ),
        
        # Gradient Boosting - Sequential ensemble model
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,       # Number of boosting stages
            max_depth=5,            # Depth of each tree (shallower than RF)
            learning_rate=0.1,      # How much each tree contributes
            min_samples_split=5,    # Minimum samples to split a node
            min_samples_leaf=2,     # Minimum samples at leaf node
            random_state=42         # For reproducibility
        )
    }
    return models


# ==============================================================================
# HYPERPARAMETER TUNING
# ==============================================================================

def get_hyperparameter_grids() -> Dict[str, Dict]:
    """
    Return hyperparameter grids for GridSearchCV tuning.
    
    These grids define the range of values to try for each hyperparameter
    during model optimization.
    
    Returns:
    --------
    Dict[str, Dict]
        Dictionary mapping model names to their hyperparameter grids
    """
    param_grids = {
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],              # Regularization strength
            'solver': ['lbfgs', 'liblinear']    # Optimization algorithms
        },
        'Random Forest': {
            'n_estimators': [50, 100, 150],     # Number of trees
            'max_depth': [10, 15, 20],          # Tree depth
            'min_samples_split': [2, 5, 10]     # Minimum samples to split
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100, 150],     # Number of boosting stages
            'max_depth': [3, 5, 7],             # Tree depth
            'learning_rate': [0.05, 0.1, 0.2]   # Contribution of each tree
        }
    }
    return param_grids


# ==============================================================================
# MODEL TRAINING
# ==============================================================================

def train_model(
    model: Any, 
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Any:
    """
    Train a machine learning model on the training data.
    
    Parameters:
    -----------
    model : Any
        Scikit-learn model instance (untrained)
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels (encoded disease values)
    
    Returns:
    --------
    Any
        Trained model ready for predictions
    """
    # Fit the model to training data
    model.fit(X_train, y_train)
    return model


# ==============================================================================
# MODEL EVALUATION
# ==============================================================================

def calculate_top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, 
                              model_classes: np.ndarray, k: int = 3) -> float:
    """
    Calculate Top-K accuracy: fraction of samples where true label is in top K predictions.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels (encoded as integers matching model_classes)
    y_proba : np.ndarray
        Probability predictions from model.predict_proba()
    model_classes : np.ndarray
        Classes from model.classes_ (defines column order of y_proba)
    k : int
        Number of top predictions to consider (default 3)
    
    Returns:
    --------
    float
        Top-K accuracy score (0.0 to 1.0)
    """
    # Get indices of top k predictions for each sample (sorted by probability)
    top_k_indices = np.argsort(y_proba, axis=1)[:, -k:]
    
    # Convert indices to class labels
    top_k_classes = model_classes[top_k_indices]
    
    # Check if true label is in top k for each sample
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_k_classes[i]:
            correct += 1
    
    return correct / len(y_true)


def calculate_roc_auc_safe(model: Any, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Optional[float]:
    """
    Calculate ROC-AUC using model.classes_ to ensure proper alignment.
    
    This function properly aligns probability predictions with labels using
    the model's classes_ attribute to avoid misalignment issues.
    
    Parameters:
    -----------
    model : Any
        Trained model with predict_proba() and classes_ attributes
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True test labels
    
    Returns:
    --------
    Optional[float]
        ROC-AUC score or None if cannot be computed
    """
    # Check if model supports probability predictions
    if not hasattr(model, 'predict_proba'):
        print("      Note: Model does not support predict_proba, ROC-AUC set to None")
        return None
    
    # Check if model has classes_ attribute
    if not hasattr(model, 'classes_'):
        print("      Note: Model missing classes_ attribute, ROC-AUC set to None")
        return None
    
    try:
        # Get probability predictions (columns match model.classes_ order)
        y_proba = model.predict_proba(X_test)
        model_classes = model.classes_
        
        # Get unique classes in test set
        test_classes = np.unique(y_test)
        
        # Check if all test classes are in model classes
        missing_classes = set(test_classes) - set(model_classes)
        if missing_classes:
            print(f"      Note: Test set has classes not seen in training, ROC-AUC set to None")
            return None
        
        # Create binarized y_test aligned with model.classes_ order
        n_samples = len(y_test)
        n_classes = len(model_classes)
        y_test_bin = np.zeros((n_samples, n_classes), dtype=int)
        
        # Create mapping from class label to column index
        class_to_idx = {c: i for i, c in enumerate(model_classes)}
        
        # Fill in the binarized matrix
        for i, label in enumerate(y_test):
            if label in class_to_idx:
                y_test_bin[i, class_to_idx[label]] = 1
        
        # Handle binary case (need 2 columns for roc_auc_score)
        if n_classes == 2:
            # For binary, use probability of positive class
            return roc_auc_score(y_test_bin[:, 1], y_proba[:, 1])
        
        # Calculate multi-class ROC-AUC (One-vs-Rest)
        return roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        
    except Exception as e:
        print(f"      Note: Could not compute ROC-AUC ({str(e)[:50]})")
        return None


def calculate_roc_auc_from_proba(y_test, y_proba: np.ndarray, 
                                  model_classes: np.ndarray) -> Optional[float]:
    """
    Calculate ROC-AUC from pre-computed probability predictions.
    
    This function takes probability outputs directly (e.g., from pipeline.predict_proba())
    and uses model_classes to ensure proper alignment for the ROC-AUC calculation.
    
    Parameters:
    -----------
    y_test : array-like
        True test labels (encoded)
    y_proba : np.ndarray
        Probability predictions (n_samples, n_classes)
    model_classes : np.ndarray
        Array of class labels in the same order as y_proba columns
    
    Returns:
    --------
    Optional[float]
        ROC-AUC score or None if cannot be computed
    """
    try:
        # Convert y_test to numpy array
        y_test_arr = np.array(y_test)
        
        # Get unique classes in test set
        test_classes = np.unique(y_test_arr)
        n_classes = len(model_classes)
        
        # Check if all test classes are in model classes
        missing_classes = set(test_classes) - set(model_classes)
        if missing_classes:
            return None
        
        # Create binarized y_test aligned with model_classes order
        n_samples = len(y_test_arr)
        y_test_bin = np.zeros((n_samples, n_classes), dtype=int)
        
        # Create mapping from class label to column index
        class_to_idx = {c: i for i, c in enumerate(model_classes)}
        
        # Fill in the binarized matrix
        for i, label in enumerate(y_test_arr):
            if label in class_to_idx:
                y_test_bin[i, class_to_idx[label]] = 1
        
        # Handle binary case
        if n_classes == 2:
            return roc_auc_score(y_test_bin[:, 1], y_proba[:, 1])
        
        # Calculate multi-class ROC-AUC (One-vs-Rest)
        return roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
        
    except Exception as e:
        return None


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    label_encoder: Any = None
) -> Dict[str, Any]:
    """
    Evaluate a trained model on test data using multiple metrics.
    
    Metrics calculated:
    - Accuracy: Percentage of correct predictions
    - Top-3 Accuracy: Percentage where true label is in top 3 predictions
    - Precision: Of predicted positives, how many are correct
    - Recall: Of actual positives, how many are predicted
    - F1 Score: Harmonic mean of precision and recall
    - ROC-AUC: Area under the ROC curve (properly aligned with model.classes_)
    
    Parameters:
    -----------
    model : Any
        Trained scikit-learn model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True test labels
    label_encoder : Any, optional
        LabelEncoder for decoding predictions
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing all evaluation metrics
    """
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Convert y_test to numpy array for consistent handling
    y_test_arr = np.array(y_test)
    
    # Calculate various evaluation metrics
    metrics = {
        # Basic accuracy (correct predictions / total)
        'accuracy': accuracy_score(y_test, y_pred),
        
        # Macro-averaged metrics (average across all classes equally)
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        
        # Weighted metrics (weighted by class frequency)
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        
        # Confusion matrix and predictions
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'predictions': y_pred
    }
    
    # Calculate Top-3 accuracy if model supports probabilities
    if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
        try:
            y_proba = model.predict_proba(X_test)
            metrics['top3_accuracy'] = calculate_top_k_accuracy(
                y_test_arr, y_proba, model.classes_, k=3
            )
        except Exception:
            metrics['top3_accuracy'] = None
    else:
        metrics['top3_accuracy'] = None
    
    # Calculate ROC-AUC using safe method with proper class alignment
    metrics['roc_auc'] = calculate_roc_auc_safe(model, X_test, y_test)
    
    return metrics


# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================

def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation on a model.
    
    Cross-validation splits data into k folds, trains on k-1 folds,
    and tests on the remaining fold. This is repeated k times.
    This gives a more robust estimate of model performance.
    
    Parameters:
    -----------
    model : Any
        Scikit-learn model instance
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    cv : int
        Number of cross-validation folds (default 5)
    
    Returns:
    --------
    Dict[str, Any]
        Cross-validation results with mean and std for each metric
    """
    # Use stratified k-fold to maintain class distribution in each fold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Calculate cross-validation scores for different metrics
    cv_results = {
        'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
        'f1_macro': cross_val_score(model, X, y, cv=skf, scoring='f1_macro'),
        'precision_macro': cross_val_score(model, X, y, cv=skf, scoring='precision_macro'),
        'recall_macro': cross_val_score(model, X, y, cv=skf, scoring='recall_macro')
    }
    
    # Calculate mean and standard deviation for each metric
    results = {}
    for metric, scores in cv_results.items():
        results[f'{metric}_mean'] = np.mean(scores)
        results[f'{metric}_std'] = np.std(scores)
        results[f'{metric}_scores'] = scores
    
    return results


# ==============================================================================
# MODEL COMPARISON
# ==============================================================================

def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: int = 5
) -> pd.DataFrame:
    """
    Train and compare all ML models.
    
    This function trains each model, evaluates on test set,
    performs cross-validation, and compiles results into a DataFrame.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test labels
    cv : int
        Number of cross-validation folds
    
    Returns:
    --------
    pd.DataFrame
        Comparison results for all models
    """
    models = get_models()
    results = []
    
    # Loop through each model
    for name, model in models.items():
        # Train model
        model = train_model(model, X_train, y_train)
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, X_test, y_test)
        
        # Perform cross-validation
        cv_metrics = cross_validate_model(model, X_train, y_train, cv)
        
        # Compile results into dictionary
        result = {
            'Model': name,
            'Test Accuracy': test_metrics['accuracy'],
            'Top-3 Accuracy': test_metrics.get('top3_accuracy'),
            'Test F1 (Macro)': test_metrics['f1_macro'],
            'Test Precision': test_metrics['precision_macro'],
            'Test Recall': test_metrics['recall_macro'],
            'CV Accuracy (Mean)': cv_metrics['accuracy_mean'],
            'CV Accuracy (Std)': cv_metrics['accuracy_std'],
            'CV F1 (Mean)': cv_metrics['f1_macro_mean'],
            'ROC-AUC': test_metrics['roc_auc']
        }
        results.append(result)
    
    return pd.DataFrame(results)


# ==============================================================================
# HYPERPARAMETER TUNING FUNCTION
# ==============================================================================

def tune_hyperparameters(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 3
) -> Tuple[Any, Dict]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    GridSearchCV tries all combinations of hyperparameters
    and selects the best combination based on cross-validation score.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to tune
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    cv : int
        Number of cross-validation folds
    
    Returns:
    --------
    Tuple[Any, Dict]
        Best model and best parameters found
    """
    models = get_models()
    param_grids = get_hyperparameter_grids()
    
    # Return default model if no grid defined
    if model_name not in param_grids:
        model = models[model_name]
        model.fit(X_train, y_train)
        return model, {}
    
    model = models[model_name]
    param_grid = param_grids[model_name]
    
    # Perform grid search
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring='f1_macro',      # Optimize for F1 score
        n_jobs=-1,               # Use all CPU cores
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_


# ==============================================================================
# FEATURE IMPORTANCE
# ==============================================================================

def get_feature_importance(
    model: Any,
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Extract feature importance from a trained model.
    
    Different models provide importance differently:
    - Random Forest & Gradient Boosting: feature_importances_ attribute
    - Logistic Regression: coefficient weights (absolute values)
    
    Parameters:
    -----------
    model : Any
        Trained model
    feature_names : List[str]
        Names of features
    top_n : int
        Number of top features to return
    
    Returns:
    --------
    pd.DataFrame
        Top N features sorted by importance
    """
    # Get feature importance based on model type
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, Gradient Boosting)
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models (Logistic Regression)
        # Use mean absolute coefficient across all classes
        importance = np.abs(model.coef_).mean(axis=0)
    else:
        # Model doesn't support feature importance
        return pd.DataFrame()
    
    # Create DataFrame and sort by importance (descending)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df.head(top_n)


# ==============================================================================
# DISEASE PREDICTION
# ==============================================================================

def predict_disease(
    model: Any,
    X: pd.DataFrame,
    label_encoder: Any,
    top_k: int = 3
) -> List[Tuple[str, float]]:
    """
    Predict disease with probability scores.
    
    Returns the top K most likely diseases with their confidence scores.
    Uses model.classes_ to properly map probability indices to class labels.
    
    Parameters:
    -----------
    model : Any
        Trained model
    X : pd.DataFrame
        Feature vector for prediction (single patient)
    label_encoder : Any
        LabelEncoder to convert numbers back to disease names
    top_k : int
        Number of top predictions to return
    
    Returns:
    --------
    List[Tuple[str, float]]
        List of (disease_name, probability) tuples
    """
    if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
        # Get probability for each disease class
        probabilities = model.predict_proba(X)[0]
        model_classes = model.classes_
        
        # Get indices of top k predictions (sorted by probability)
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        # Convert indices to disease names using model.classes_ for proper mapping
        predictions = []
        for idx in top_indices:
            class_label = model_classes[idx]
            disease_name = label_encoder.inverse_transform([class_label])[0]
            predictions.append((disease_name, probabilities[idx]))
        
    elif hasattr(model, 'predict_proba'):
        # Fallback: use label_encoder order (less reliable)
        probabilities = model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        predictions = [
            (label_encoder.inverse_transform([idx])[0], probabilities[idx])
            for idx in top_indices
        ]
    else:
        # If model doesn't support probabilities, just get single prediction
        prediction = model.predict(X)[0]
        disease = label_encoder.inverse_transform([prediction])[0]
        predictions = [(disease, 1.0)]
    
    return predictions
