import joblib
import os
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_stacking_ensemble():
    """
    Builds and returns a StackingClassifier
    """
    et_path = "models/extratrees_model.joblib"
    lgbm_path = "models/optimized_lightgbm_model.joblib"
    xgb_path = "models/optimized_xgboost_model.joblib"
    hgb_path = "models/optimized_hgboost_model.joblib"

    et_model = joblib.load(et_path)
    lgbm_model = joblib.load(lgbm_path)
    xgb_model = joblib.load(xgb_path)
    hgb_model = joblib.load(hgb_path)

    base_learners = [
        ("extratrees", et_model),
        ("lightgbm", lgbm_model),
        ("xgboost", xgb_model),
        ("hgboost", hgb_model)
    ]

    final_estimator = LogisticRegression(
        solver='lbfgs',
        penalty='l2',
        max_iter=1000,
        random_state=42
    )

    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=final_estimator,
        cv=10,
        passthrough=False,
        n_jobs=-1
    )

    return stacking_clf
