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

- `main.py` – Entry point for training and evaluation  
- `src/` – Core modules (preprocessing, feature engineering, models, pipeline)  
- `requirements.txt` – Project dependencies  
- `data/` – Dataset folder (not included in repository)  

---

## Dataset

This project uses the Kaggle dataset:

**Healthcare Symptoms Disease Classification Dataset**

Source:  
https://www.kaggle.com/datasets/kundanbedmutha/healthcare-symptomsdisease-classification-dataset

The dataset is not included in this repository due to Kaggle licensing terms.

### How to use the dataset

1. Download the dataset from Kaggle  
2. Extract the dataset file  
3. Place the file in:

---

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
python main.py
