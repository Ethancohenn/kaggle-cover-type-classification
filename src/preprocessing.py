import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from src.feature_engineering import engineer_features

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def split_features_and_target(train_df, target_column='Cover_Type'):
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    return X, y


def train_val_split(X, y, test_size=0.1, random_state=42):
    """
    Split data into training and validation sets
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def rebalance_classes(X_train, y_train):
    """
    Apply resampling strategy to rebalance class distribution in the training set
    """
    ros = RandomOverSampler(sampling_strategy={1: y_train.value_counts()[1] * 2,
                                                2: y_train.value_counts()[2] * 3})
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def preprocess_pipeline(train_path="data/train.csv", test_path="data/test.csv"):
    train_df, test_df = load_data(train_path, test_path)
    train_df = engineer_features(train_df)
    test_df = engineer_features(test_df)
    X, y = split_features_and_target(train_df)
    X_train, X_val, y_train, y_val = train_val_split(X, y)
    X_train_bal, y_train_bal = rebalance_classes(X_train, y_train)
    return X_train_bal, X_val, y_train_bal, y_val, test_df

