from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        criterion="gini",
        max_features="sqrt",
        max_depth=None,
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    n_classes = len(np.unique(y_train_encoded))
    model = XGBClassifier(
        eval_metric="mlogloss",
        n_estimators=150,
        random_state=42, 
        num_class=n_classes
    )
    model.fit(X_train, y_train_encoded)
    model.label_encoder = label_encoder
    return model


def train_histogram_gb(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    model = HistGradientBoostingClassifier(
        max_iter=250,
        random_state=42
    )
    model.fit(X_train, y_train_encoded)
    return model, label_encoder


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=10,
        loss_function='MultiClass',
        eval_metric='Accuracy',
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train_encoded)
    return model


def train_extra_trees(X_train, y_train):
    model = ExtraTreesClassifier(
        n_estimators=150,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model



MODELS = {
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "histogram_gb": train_histogram_gb,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
    "extra_trees": train_extra_trees
}
