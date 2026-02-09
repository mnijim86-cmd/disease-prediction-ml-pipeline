"""
================================================================================
Disease Prediction using Machine Learning
================================================================================
A traditional ML approach to healthcare diagnostics.

This script trains and evaluates 3 machine learning models for disease prediction
based on patient symptoms, age, and gender.

Models used:
- Logistic Regression: Linear baseline with interpretable coefficients
- Random Forest: Ensemble method capturing non-linear relationships
- Gradient Boosting: Sequential tree building for strong performance

Usage:
    python main.py                      # Use full dataset with Pipeline
    python main.py --max_rows 5000      # Use stratified sample of 5000 rows
    python main.py --data path/to.csv   # Use custom data file

Author: Mahmoud Nijim
Course: Machine Learning
================================================================================
"""

# ==============================================================================
# IMPORT REQUIRED LIBRARIES
# ==============================================================================
import argparse
import os
import sys
import pandas as pd
import numpy as np
import joblib

# Get script directory for relative path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Import our custom modules
from src.data_preprocessing import load_data, clean_data, get_data_summary
from src.models import get_models, get_feature_importance, calculate_top_k_accuracy
from src.pipeline import (
    create_disease_pipeline,
    get_feature_names_from_pipeline,
    get_symptom_list_from_pipeline
)


# ==============================================================================
# COMMAND LINE ARGUMENT PARSING
# ==============================================================================

def parse_arguments():
    """
    Parse command line arguments for the disease prediction script.
    
    Arguments:
    - --data: Path to the dataset CSV file (default: data/Healthcare.csv)
    - --max_rows: Maximum number of rows to use (stratified sampling)
    - --cv_folds: Number of cross-validation folds (default: 5)
    - --no_save: Disable saving model artifacts
    
    Returns:
    --------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Disease Prediction using Machine Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                          # Use full dataset
    python main.py --max_rows 5000          # Stratified sample of 5000 rows
    python main.py --data my_data.csv       # Use custom dataset
    python main.py --cv_folds 10            # Use 10-fold cross-validation
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default=os.path.join(SCRIPT_DIR, 'data', 'Healthcare.csv'),
        help='Path to the healthcare dataset CSV file'
    )
    
    parser.add_argument(
        '--max_rows', 
        type=int, 
        default=None,
        help='Maximum rows to use (stratified sampling). If not set, uses full dataset.'
    )
    
    parser.add_argument(
        '--cv_folds', 
        type=int, 
        default=5,
        help='Number of cross-validation folds (default: 5)'
    )
    
    parser.add_argument(
        '--no_save', 
        action='store_true',
        help='Disable saving model artifacts'
    )
    
    return parser.parse_args()


# ==============================================================================
# HELPER FUNCTIONS FOR DISPLAYING OUTPUT
# ==============================================================================

def print_separator(title: str = ""):
    """
    Print a formatted separator line with optional title.
    This helps organize the console output into readable sections.
    """
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_data_summary(df: pd.DataFrame):
    """
    Display dataset summary statistics.
    Shows total samples, features, diseases, symptoms, and distributions.
    """
    print_separator("DATASET SUMMARY")
    
    # Get summary statistics from our preprocessing module
    summary = get_data_summary(df)
    
    # Display basic counts
    print(f"\nTotal Samples: {summary['total_samples']:,}")
    print(f"Number of Features: {summary['num_features']}")
    print(f"Number of Diseases: {summary['num_diseases']}")
    print(f"Number of Unique Symptoms: {summary['num_unique_symptoms']}")
    
    # Display age statistics (min, max, mean, median)
    print(f"\nAge Statistics:")
    print(f"  - Min: {summary['age_stats']['min']:.0f}")
    print(f"  - Max: {summary['age_stats']['max']:.0f}")
    print(f"  - Mean: {summary['age_stats']['mean']:.1f}")
    print(f"  - Median: {summary['age_stats']['median']:.0f}")
    
    # Display gender distribution
    print(f"\nGender Distribution:")
    for gender, count in summary['gender_distribution'].items():
        print(f"  - {gender}: {count:,}")
    
    # Display symptom count statistics
    print(f"\nSymptom Count Statistics:")
    print(f"  - Min: {summary['symptom_count_stats']['min']}")
    print(f"  - Max: {summary['symptom_count_stats']['max']}")
    print(f"  - Mean: {summary['symptom_count_stats']['mean']:.1f}")
    
    # Display top 10 most common diseases
    print(f"\nTop 10 Diseases:")
    sorted_diseases = sorted(summary['disease_distribution'].items(), 
                            key=lambda x: x[1], reverse=True)[:10]
    for disease, count in sorted_diseases:
        print(f"  - {disease}: {count:,}")


# ==============================================================================
# PIPELINE TRAINING AND EVALUATION
# ==============================================================================

def evaluate_pipeline(pipeline, X_test, y_test):
    """
    Evaluate a trained pipeline on test data.
    
    Returns metrics including accuracy, Top-3 accuracy, and ROC-AUC.
    Uses pipeline.predict_proba() directly for safer probability calculation.
    """
    # Get the classifier from the pipeline
    classifier = pipeline.named_steps['classifier']
    
    # Make predictions using the full pipeline
    y_pred = pipeline.predict(X_test)
    
    # Calculate basic metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    
    # Calculate Top-3 accuracy and ROC-AUC using pipeline.predict_proba() directly
    if hasattr(pipeline, 'predict_proba'):
        try:
            # Use pipeline.predict_proba() directly - safer than manual transforms
            y_proba = pipeline.predict_proba(X_test)
            
            # Get model classes from classifier for proper alignment
            model_classes = classifier.classes_
            
            # Top-3 accuracy
            metrics['top3_accuracy'] = calculate_top_k_accuracy(
                np.array(y_test), y_proba, model_classes, k=3
            )
            
            # ROC-AUC using the new helper function with proper class alignment
            from src.models import calculate_roc_auc_from_proba
            metrics['roc_auc'] = calculate_roc_auc_from_proba(y_test, y_proba, model_classes)
        except Exception as e:
            print(f"      Note: Could not compute advanced metrics ({str(e)[:30]})")
            metrics['top3_accuracy'] = None
            metrics['roc_auc'] = None
    else:
        metrics['top3_accuracy'] = None
        metrics['roc_auc'] = None
    
    return metrics


def train_and_evaluate_pipelines(df_train, y_train, df_test, y_test, cv_folds=5):
    """
    Train all 3 ML models using sklearn Pipeline and evaluate their performance.
    
    Using Pipeline ensures:
    1. No data leakage during cross-validation
    2. Consistent feature transformation for training and prediction
    3. Easy model serialization
    """
    print_separator("MODEL TRAINING & EVALUATION (Pipeline)")
    
    # Get dictionary of all models to train
    models_dict = get_models()
    results = []
    trained_pipelines = {}
    
    # Determine which models need scaling (Logistic Regression benefits from it)
    scale_models = {'Logistic Regression'}
    
    # Loop through each model and train/evaluate it
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        
        # Create pipeline with feature transformer
        scale = name in scale_models
        pipeline = create_disease_pipeline(model, scale_features=scale)
        
        # Train the pipeline on training data
        pipeline.fit(df_train, y_train)
        trained_pipelines[name] = pipeline
        
        # Evaluate on test set
        test_metrics = evaluate_pipeline(pipeline, df_test, y_test)
        
        # Perform k-fold cross-validation (CV happens inside pipeline - no leakage!)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, df_train, y_train, cv=skf, scoring='accuracy')
        
        # Store all metrics in a dictionary
        result = {
            'Model': name,
            'Test Accuracy': test_metrics['accuracy'],
            'Top-3 Accuracy': test_metrics.get('top3_accuracy'),
            'Test F1 (Macro)': test_metrics['f1_macro'],
            'Test Precision': test_metrics['precision_macro'],
            'Test Recall': test_metrics['recall_macro'],
            'CV Accuracy (Mean)': np.mean(cv_scores),
            'CV Accuracy (Std)': np.std(cv_scores),
            'ROC-AUC': test_metrics.get('roc_auc')
        }
        results.append(result)
        
        # Print results for this model
        print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
        top3 = test_metrics.get('top3_accuracy')
        if top3 is not None:
            print(f"  Top-3 Accuracy: {top3:.4f}")
        print(f"  Test F1 Score: {test_metrics['f1_macro']:.4f}")
        print(f"  CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        if test_metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
    
    # Return results as DataFrame and dictionary of trained pipelines
    return pd.DataFrame(results), trained_pipelines


def print_model_comparison(results_df: pd.DataFrame):
    """
    Print formatted model comparison table.
    Shows all metrics for each model and highlights the best performer.
    """
    print_separator("MODEL COMPARISON RESULTS")
    
    # Print the results table
    print("\n" + results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    
    # Find and display the best model based on Test Accuracy
    best_model = results_df.loc[results_df['Test Accuracy'].idxmax(), 'Model']
    best_accuracy = results_df['Test Accuracy'].max()
    
    print(f"\n{'*' * 50}")
    print(f"  BEST MODEL: {best_model}")
    print(f"  Test Accuracy: {best_accuracy:.4f}")
    print(f"{'*' * 50}")
    
    return best_model


def print_feature_importance(trained_pipelines: dict, top_n: int = 15):
    """
    Print feature importance for models that support it.
    Extracts feature names from the pipeline's FeatureTransformer.
    """
    print_separator("FEATURE IMPORTANCE")
    
    # Loop through each trained pipeline
    for name, pipeline in trained_pipelines.items():
        # Get feature names from the pipeline
        feature_names = get_feature_names_from_pipeline(pipeline)
        
        # Get the classifier
        classifier = pipeline.named_steps['classifier']
        
        # Get feature importance
        importance_df = get_feature_importance(classifier, feature_names, top_n)
        
        # Print if the model supports feature importance
        if not importance_df.empty:
            print(f"\n{name} - Top {top_n} Features:")
            print("-" * 40)
            for i, row in importance_df.iterrows():
                # Clean up feature name for display
                feature_name = row['Feature'].replace('symptom_', '').replace('_', ' ')
                print(f"  {feature_name:30s} {row['Importance']:.4f}")


# ==============================================================================
# ARTIFACT SAVING FUNCTIONS
# ==============================================================================

def save_artifacts(best_model_name, trained_pipelines, label_encoder):
    """
    Save the best model pipeline and vocabulary artifacts to disk.
    
    Saves:
    - models/best_model.joblib: The best performing pipeline (includes transformer)
    - models/vocabulary.joblib: Label encoder and metadata
    - models/all_pipelines.joblib: All trained pipelines
    """
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the best pipeline (includes FeatureTransformer + model)
    best_pipeline = trained_pipelines[best_model_name]
    model_path = 'models/best_model.joblib'
    joblib.dump(best_pipeline, model_path)
    print(f"      Best model ({best_model_name}) saved to '{model_path}'")
    
    # Get symptom list and feature names from the pipeline
    symptom_list = get_symptom_list_from_pipeline(best_pipeline)
    feature_columns = get_feature_names_from_pipeline(best_pipeline)
    
    # Save vocabulary and metadata
    vocabulary = {
        'symptom_list': symptom_list,
        'label_encoder': label_encoder,
        'feature_columns': feature_columns,
        'best_model_name': best_model_name
    }
    vocab_path = 'models/vocabulary.joblib'
    joblib.dump(vocabulary, vocab_path)
    print(f"      Vocabulary saved to '{vocab_path}'")
    
    # Also save all trained pipelines for comparison
    all_pipelines_path = 'models/all_pipelines.joblib'
    joblib.dump(trained_pipelines, all_pipelines_path)
    print(f"      All pipelines saved to '{all_pipelines_path}'")


def load_artifacts():
    """
    Load saved pipeline artifacts from disk.
    
    Returns:
    --------
    Tuple containing:
        - pipeline: The best trained pipeline
        - vocabulary: Dictionary with label_encoder and metadata
    """
    pipeline = joblib.load('models/best_model.joblib')
    vocabulary = joblib.load('models/vocabulary.joblib')
    return pipeline, vocabulary


# ==============================================================================
# INTERACTIVE PREDICTION FUNCTION
# ==============================================================================

def interactive_prediction(trained_pipelines, label_encoder):
    """
    Interactive disease prediction based on user input.
    
    Uses the Pipeline directly for prediction, which handles
    all feature transformation automatically.
    """
    print_separator("DISEASE PREDICTION")
    
    # Let user select a model
    print("\nSelect a model for prediction:")
    model_names = list(trained_pipelines.keys())
    for i, name in enumerate(model_names, 1):
        print(f"  {i}. {name}")
    
    # Get model choice from user with error handling
    try:
        model_choice = int(input("\nEnter model number (1-3): ")) - 1
        if model_choice < 0 or model_choice >= len(model_names):
            print("Invalid choice. Using Random Forest.")
            model_choice = 1
    except ValueError:
        print("Invalid input. Using Random Forest.")
        model_choice = 1
    
    selected_pipeline = trained_pipelines[model_names[model_choice]]
    print(f"\nUsing: {model_names[model_choice]}")
    
    # Get symptom list from the pipeline
    symptom_list = get_symptom_list_from_pipeline(selected_pipeline)
    
    # Get patient age with error handling
    try:
        age = int(input("\nEnter patient age: "))
    except ValueError:
        print("Invalid age. Using 30.")
        age = 30
    
    # Get patient gender
    print("\nEnter gender (Male/Female/Other): ", end="")
    gender = input().strip().title()
    if gender not in ['Male', 'Female', 'Other']:
        print("Invalid gender. Using 'Other'.")
        gender = 'Other'
    
    # Display available symptoms for user reference
    print("\nAvailable symptoms:")
    formatted_symptoms = [s.replace('_', ' ').title() for s in symptom_list]
    for i in range(0, len(formatted_symptoms), 5):
        row = formatted_symptoms[i:i+5]
        print("  " + ", ".join(row))
    
    # Get symptoms from user
    print("\nEnter symptoms (comma-separated): ", end="")
    symptom_input = input().strip()
    selected_symptoms = [s.strip().lower() for s in symptom_input.split(',') if s.strip()]
    
    # Validate that at least one symptom was entered
    if not selected_symptoms:
        print("\nNo symptoms entered. Please enter at least one symptom.")
        return
    
    # Create a DataFrame for prediction (Pipeline expects DataFrame)
    patient_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Symptoms': [', '.join(selected_symptoms)],
        'Symptom_Count': [len(selected_symptoms)]
    })
    
    # Get the classifier from pipeline for class labels
    classifier = selected_pipeline.named_steps['classifier']
    
    # Make prediction using pipeline.predict_proba() directly
    if hasattr(selected_pipeline, 'predict_proba'):
        # Use pipeline.predict_proba() directly - safer approach
        probabilities = selected_pipeline.predict_proba(patient_data)[0]
        
        # Get top 3 predictions using classifier.classes_ for alignment
        top_indices = np.argsort(probabilities)[-3:][::-1]
        predictions = []
        for idx in top_indices:
            class_label = classifier.classes_[idx]
            disease_name = label_encoder.inverse_transform([class_label])[0]
            predictions.append((disease_name, probabilities[idx]))
    else:
        # If no probabilities, just get single prediction
        pred = selected_pipeline.predict(patient_data)[0]
        disease_name = label_encoder.inverse_transform([pred])[0]
        predictions = [(disease_name, 1.0)]
    
    # Display prediction results
    print("\n" + "-" * 40)
    print("PREDICTION RESULTS")
    print("-" * 40)
    print(f"\nPatient Information:")
    print(f"  Age: {age}")
    print(f"  Gender: {gender}")
    print(f"  Symptoms: {', '.join(selected_symptoms)}")
    
    print(f"\nTop 3 Predicted Diseases:")
    for i, (disease, prob) in enumerate(predictions, 1):
        print(f"  {i}. {disease} - Confidence: {prob*100:.1f}%")
    
    # Display disclaimer
    print("\n" + "*" * 50)
    print("  DISCLAIMER: This prediction is for educational")
    print("  purposes only. Consult a healthcare professional.")
    print("*" * 50)


# ==============================================================================
# MAIN FUNCTION - ENTRY POINT
# ==============================================================================

def main():
    """
    Main function to run the complete disease prediction ML pipeline.
    
    Pipeline steps:
    1. Parse command line arguments
    2. Load and clean data
    3. Apply stratified sampling if --max_rows specified
    4. Encode target labels
    5. Split data into train/test sets
    6. Train and evaluate pipelines (no data leakage!)
    7. Save artifacts (pipeline and vocabulary)
    8. Offer interactive prediction
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Display program header
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#    DISEASE PREDICTION USING MACHINE LEARNING" + " " * 22 + "#")
    print("#    Traditional ML Approach (sklearn Pipeline)" + " " * 22 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Display configuration
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data}")
    print(f"  Max rows: {'Full dataset' if args.max_rows is None else args.max_rows}")
    print(f"  CV folds: {args.cv_folds}")
    print(f"  Save model: {not args.no_save}")
    
    # -------------------------------------------------------------------------
    # STEP 1: Load and clean the dataset
    # -------------------------------------------------------------------------
    print("\n[1/6] Loading and cleaning data...")
    df = load_data(args.data)
    df = clean_data(df)
    print(f"      Loaded {len(df):,} records")
    
    # -------------------------------------------------------------------------
    # STEP 2: Apply stratified sampling if max_rows specified
    # -------------------------------------------------------------------------
    if args.max_rows is not None and args.max_rows < len(df):
        print(f"\n[2/6] Applying stratified sampling to {args.max_rows:,} rows...")
        
        try:
            # Use stratified sampling to preserve class distribution
            df_sampled, _ = train_test_split(
                df, 
                train_size=args.max_rows, 
                random_state=42, 
                stratify=df['Disease']
            )
            df = df_sampled.reset_index(drop=True)
            print(f"      Sampled {len(df):,} records (stratified by disease)")
        except ValueError as e:
            # If stratified sampling fails (e.g., too few samples per class),
            # fall back to random sampling
            print(f"      Warning: Stratified sampling failed ({str(e)[:50]})")
            print(f"      Falling back to random sampling...")
            df_sampled = df.sample(n=args.max_rows, random_state=42)
            df = df_sampled.reset_index(drop=True)
            print(f"      Sampled {len(df):,} records (random sampling)")
    else:
        print(f"\n[2/6] Using full dataset ({len(df):,} records)")
    
    # Display dataset summary
    print_data_summary(df)
    
    # -------------------------------------------------------------------------
    # STEP 3: Encode target labels
    # -------------------------------------------------------------------------
    print("\n[3/6] Encoding target labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Disease'])
    print(f"      Encoded {len(label_encoder.classes_)} disease classes")
    
    # -------------------------------------------------------------------------
    # STEP 4: Split data into training and test sets
    # -------------------------------------------------------------------------
    print("\n[4/6] Splitting data...")
    
    # Split into 80% training and 20% test (stratified to maintain class balance)
    # Note: We pass the DataFrame, not extracted features - Pipeline handles transformation
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training set: {len(df_train):,} samples")
    print(f"      Test set: {len(df_test):,} samples")
    
    # -------------------------------------------------------------------------
    # STEP 5: Train and evaluate all pipelines
    # -------------------------------------------------------------------------
    print("\n[5/6] Training and evaluating pipelines...")
    results_df, trained_pipelines = train_and_evaluate_pipelines(
        df_train, y_train, df_test, y_test, cv_folds=args.cv_folds
    )
    
    # Display model comparison results and get best model name
    best_model_name = print_model_comparison(results_df)
    
    # Display feature importance for each model
    print_feature_importance(trained_pipelines, top_n=15)
    
    # -------------------------------------------------------------------------
    # STEP 6: Save results and artifacts
    # -------------------------------------------------------------------------
    print("\n[6/6] Saving results...")
    results_df.to_csv('model_results.csv', index=False)
    print("      Results saved to 'model_results.csv'")
    
    # Save pipeline artifacts if not disabled
    if not args.no_save:
        save_artifacts(best_model_name, trained_pipelines, label_encoder)
    
    # Display completion message
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # INTERACTIVE PREDICTION LOOP (skip if non-interactive environment)
    # -------------------------------------------------------------------------
    if not sys.stdin.isatty():
        print("\nNon-interactive environment detected. Skipping interactive menu.")
        print("Thank you for using the Disease Prediction System!")
        return
    
    try:
        while True:
            print("\nOptions:")
            print("  1. Make a disease prediction")
            print("  2. Exit")
            
            choice = input("\nEnter choice (1 or 2): ").strip()
            
            if choice == '1':
                interactive_prediction(trained_pipelines, label_encoder)
            elif choice == '2':
                print("\nThank you for using the Disease Prediction System!")
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
    except (EOFError, KeyboardInterrupt):
        print("\n\nExiting. Thank you for using the Disease Prediction System!")


# ==============================================================================
# PROGRAM ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    main()
