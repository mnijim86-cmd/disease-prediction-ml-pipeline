# Disease Prediction – End-to-End ML Pipeline

An end-to-end machine learning pipeline built using Python and scikit-learn.

This project focuses on building a structured, reproducible, and leakage-safe machine learning workflow for disease classification based on healthcare symptoms data.

---

## Overview

This project implements a clean ML pipeline including:

- Data preprocessing and feature engineering  
- Stratified cross-validation  
- Model comparison (Logistic Regression, Random Forest, Gradient Boosting)  
- Evaluation using macro-averaged metrics  
- Structured and reproducible experimentation  

The goal is to prevent data leakage and ensure reliable model evaluation.

---

## Project Structure

```text
disease-prediction-ml-pipeline/
│
├── main.py
├── requirements.txt
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── pipeline.py
│
└── data/ (not included in repository)
```

---

## Dataset

This project uses the Kaggle dataset:

**Healthcare Symptoms Disease Classification Dataset**

🔗 Source:  
https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset

The dataset is **not included** in this repository due to Kaggle licensing terms.

### How to Use the Dataset

1. Download the dataset from Kaggle  
2. Extract the dataset files  
3. Place the dataset file inside:

```text
data/Healthcare.csv
```

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
python main.py
```

---

## Technologies Used

- Python  
- scikit-learn  
- pandas  
- NumPy  
- Classical Machine Learning models  
- Stratified Cross-Validation  

---

## Key Focus Areas

- Preventing data leakage  
- Proper validation methodology  
- Reproducible ML experimentation  
- Structured project architecture  

---

## Author

Mahmoud Z A Nijim  
Master’s Student in Computer Science – Blekinge Institute of Technology  
AI / Machine Learning Focus


