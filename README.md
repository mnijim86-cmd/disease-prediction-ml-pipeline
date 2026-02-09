# Disease Prediction – End-to-End ML Pipeline

An end-to-end machine learning pipeline built using Python and scikit-learn.

## Overview

This project implements a structured and reproducible ML workflow including:

- Data preprocessing and feature engineering
- Stratified cross-validation
- Model comparison (Logistic Regression, Random Forest, Gradient Boosting)
- Evaluation using macro-averaged metrics
- Clean experiment structure

The goal is to prevent data leakage and ensure reliable evaluation.

## Project Structure

- main.py – Entry point for training and evaluation
- src/ – Core modules (preprocessing, feature engineering, models, pipeline)
- requirements.txt – Project dependencies
- data/ – Dataset folder (not included in repository)

## Installation

```bash
pip install -r requirements.txt
python main.py