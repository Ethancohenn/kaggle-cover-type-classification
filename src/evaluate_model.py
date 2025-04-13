import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")


def evaluate_model(model, X_val, y_val, label_encoder=None, model_name=""):
    """
    Generate and display classification report for a given model.
    """
    if label_encoder:
        y_val_encoded = label_encoder.fit_transform(y_val)
        y_pred = model.predict(X_val)
        y_pred = np.array(y_pred)
    else:
        y_val_encoded = y_val
        y_pred = model.predict(X_val)
    
    report = classification_report(y_val_encoded, y_pred, output_dict=True)
    print(f"\n Classification Report for {model_name}:\n")
    print(classification_report(y_val_encoded, y_pred))

    return report


def cross_validate_model(model, X_train, y_train, cv=5, scoring='accuracy'):
    """
    Perform cross-validation and return mean accuracy.
    """
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
    print(f"Mean CV {scoring}: {scores.mean():.4f}")
    return scores
