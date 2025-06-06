import numpy as np
import pandas as pd
from enum import Enum
from sklearn.model_selection import train_test_split

class DatasetDirectory(Enum):
    HEART_DISEASE = "heart+disease/processed.cleveland.data"
    PENGUINS = "penguins/penguins.csv"
    BANK = "bank/bank.csv"


def load_data(dataset_directory: DatasetDirectory):
    if dataset_directory == DatasetDirectory.BANK:
        data = pd.read_csv(dataset_directory.value, header=0, sep=";")
    else:
        data = pd.read_csv(dataset_directory.value, header=0)
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    return data


def heart_disease_data_preprocessing(data):
    data.columns = [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
        ]
    data["target"] = data["target"].replace([2, 3, 4], 1)


def penguins_data_preprocessing(data):
    data.columns = [
            "species", "island", "bill_length_mm",
            "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex", "year"
        ]
    data["species"] = data["species"].map({
        "Adelie": 0,
        "Chinstrap": 1,
        "Gentoo": 2
    })
    categorical_cols = data.select_dtypes(include=['object']).columns.drop('species', errors='ignore')
    data_encoded = pd.get_dummies(data, columns=categorical_cols)
    # Cập nhật lại data
    data.drop(data.index, inplace=True)
    for col in data_encoded.columns:
        data[col] = data_encoded[col]
    
    data.dropna(axis=1,how='all', inplace=True)  # Loại bỏ các cột có toàn giá trị NaN
    
    


def bank_data_preprocessing(data):
    data.columns = [
            "age", "job", "marital", "education", "default", "balance",
            "housing", "loan", "contact", "day", "month", "duration",
            "campaign", "pdays", "previous", "poutcome", "y"
        ]
    data["y"] = data["y"].map({'yes': 1, 'no': 0})
    # One-hot encoding cho các cột dạng object (chuỗi), loại bỏ cột 'y'
    categorical_cols = data.select_dtypes(include=['object']).columns.drop('y', errors='ignore')
    data_encoded = pd.get_dummies(data, columns=categorical_cols)
    # Cập nhật lại data
    data.drop(data.index, inplace=True)
    for col in data_encoded.columns:
        data[col] = data_encoded[col]

    data.dropna(axis=1,how='all', inplace=True)  # Loại bỏ các cột có toàn giá trị NaN


def data_preprocessing(data, dataset_directory: DatasetDirectory):
    if dataset_directory == DatasetDirectory.HEART_DISEASE:
        heart_disease_data_preprocessing(data)
    elif dataset_directory == DatasetDirectory.PENGUINS:
        penguins_data_preprocessing(data)
    elif dataset_directory == DatasetDirectory.BANK:
        bank_data_preprocessing(data)


def prepare_dataset_v2(data, train_ratio, test_ratio, dataset_directory: DatasetDirectory):
    if dataset_directory == DatasetDirectory.HEART_DISEASE:
        X = data.drop(columns=["target"]).values
        y = data["target"].values
        feature_train, feature_test, label_train, label_test = train_test_split(
            X, y, train_size=train_ratio, test_size=test_ratio, random_state=42
        )
        return feature_train, label_train, feature_test, label_test
    elif dataset_directory == DatasetDirectory.PENGUINS:
        X = data.drop(columns=["species"]).values
        y = data["species"].values
        feature_train, feature_test, label_train, label_test = train_test_split(
            X, y, train_size=train_ratio, test_size=test_ratio, random_state=42
        )
        return feature_train, label_train, feature_test, label_test
    elif dataset_directory == DatasetDirectory.BANK:
        X = data.drop(columns=["y"]).values
        y = data["y"].values
        feature_train, feature_test, label_train, label_test = train_test_split(
            X, y, train_size=train_ratio, test_size=test_ratio, random_state=42
        )
        return feature_train, label_train, feature_test, label_test