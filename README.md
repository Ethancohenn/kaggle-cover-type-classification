# Predicting Forest Cover Type - Kaggle Challenge

### Project Overview

This repository contains the implementation and detailed analysis performed for the Kaggle challenge **"Forest Cover Type Prediction"**, carried out as part of the course **APM-51053-EP - Foundations of Machine Learning**.

**Contributors**: Ethan Cohen, Arthur Iffenecker, Tanguy Azema, Jules Cognon

### Objective

The project's goal was to classify forest cover types based on geographical and environmental data. The dataset contained 55 diverse features, including numeric (elevation, slope) and binary categorical variables (soil type, wilderness area).

### Dataset

The dataset is sourced from the Kaggle competition:
- [Forest Cover Type Prediction - Kaggle](https://www.kaggle.com/competitions/forest-3-a-2024)

The provided data included:
- `train.csv`: Training data with 55 features
- `test-full.csv`: Dataset for predictions
- `full-submission.csv`: Submission example format

### Key Techniques

#### Data Processing and Feature Engineering:
- Exploratory Data Analysis (EDA)
- Feature creation (combined distances, climatic and geological zones, ...)
- Handling imbalanced data using RandomOverSampler

#### Models Implemented:

- **Simple Models**: K-Nearest Neighbors (KNN), Decision Trees
- **Bagging Algorithms**: Random Forest, Extra Trees
- **Boosting Algorithms**: XGBoost, LightGBM, Histogram Gradient Boosting, CatBoost
- **Neural Networks**: Multilayer perceptron
- **Ensemble Methods**: Voting Classifier (soft voting) and Stacking

#### Hyperparameter Optimization:
- GridSearchCV
- RandomizedSearchCV
- Bayesian Optimization (BayesSearchCV, Optuna)

### Results

The best-performing model achieved an accuracy of **0.84241** using **Histogram Gradient Boosting**, optimized via Bayesian optimization techniques.

### Future Work
- Improve feature engineering, especially for distinguishing Cover Types 1 and 2.
- Experiment with advanced methods for imbalanced datasets.
- Explore deep learning methods with more sophisticated architectures or data preprocessing strategies.

### How to Use

Clone the repository:
```bash
git clone [REPO_URL]
cd [REPO_NAME]
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Explore notebooks for step-by-step analysis and model training:
```bash
jupyter notebook
```
---

This project provided valuable insights into various machine learning methods, emphasizing the importance of meticulous hyperparameter tuning and model selection.

