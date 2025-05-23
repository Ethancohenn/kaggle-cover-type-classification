# Predicting Forest Cover Type - Kaggle Challenge

### Project Overview

This repository contains the implementation and detailed analysis performed for the Kaggle challenge **"Forest Cover Type Prediction"**, carried out as part of the course **APM-51053-EP - Foundations of Machine Learning**.

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
git clone https://github.com/Ethancohenn/kaggle-cover-type-classification.git
cd kaggle-cover-type-classification
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

**Note**: Some parts of the analysis, such as neural networks, were explored in the final report ([Cover Type Classification Final Report.pdf](report/Cover%20Type%20Classification%20Final%20Report.pdf)) but are not included in this code repository.

