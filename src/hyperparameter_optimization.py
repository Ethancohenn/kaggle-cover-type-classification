from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
import optuna

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import numpy as np

# RANDOM FOREST
def optimize_random_forest(X, y, method='grid'):
    param_grid = {
        "n_estimators": [100, 150, 200],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"],
        "criterion": ["gini", "entropy"]
    }
    model = RandomForestClassifier(random_state=42)

    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    elif method == 'bayesian':
        search = BayesSearchCV(model, param_grid, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'grid', 'random', 'bayesian'.")
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

# LIGHTGBM
def optimize_lightgbm(X, y, method='grid'):
    param_grid = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.05, 0.1],
        "num_leaves": [31, 63, 127]
    }
    model = LGBMClassifier(random_state=42)

    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    elif method == 'bayesian':
        search = BayesSearchCV(model, param_grid, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'grid', 'random', 'bayesian'.")
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

# XGBOOST + OPTUNA
def optimize_xgboost_with_optuna(X, y, n_trials=10):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10)
        }
        model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        model.fit(X, y_encoded)
        y_pred = model.predict(X)
        return accuracy_score(y_encoded, y_pred)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_model = XGBClassifier(**study.best_params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    best_model.fit(X, y_encoded)
    return best_model, study.best_params

# HISTOGRAM GRADIENT BOOSTING
def optimize_histgb(X, y, method='grid'):
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'max_leaf_nodes': [31, 63, 127],
        'max_iter': [100, 200, 300],
        'l2_regularization': [0, 1, 10],
        'max_bins': [32, 64, 128, 255]
    }
    model = HistGradientBoostingClassifier(random_state=42)

    if method == 'grid':
        search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    elif method == 'random':
        search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    elif method == 'bayesian':
        search = BayesSearchCV(model, param_grid, n_iter=20, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    else:
        raise ValueError("Invalid method. Choose from 'grid', 'random', 'bayesian'.")
    
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


