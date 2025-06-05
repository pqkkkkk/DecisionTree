import numpy as np
import pandas as pd
from enum import Enum
from sklearn.tree import DecisionTreeClassifier

class DatasetDirectory(Enum):
    HEART_DISEASE = "heart+disease/processed.cleveland.data"
    PENGUINS = "penguins/penguins.csv"
def load_data(dataset_directory: DatasetDirectory):
    data = pd.read_csv(dataset_directory.value, header=None)
    data.replace('?', np.nan, inplace=True)
    data.dropna()
    return data
def data_preprocessing(data, dataset_directory: DatasetDirectory):
    if dataset_directory == DatasetDirectory.HEART_DISEASE:
        # Chuyển tất cả giá trị ở cột cuối cùng thành 1 nếu nó bằng 2, 3 hoặc 4
        data.iloc[:, -1] = data.iloc[:, -1].replace([2, 3, 4], 1)
    elif dataset_directory == DatasetDirectory.PENGUINS:
        pass
def prepare_dataset(data, train_ratio, test_ratio):
    data_array = data.to_numpy()

    np.random.shuffle(data_array)

    train_data_size = int(len(data_array) * train_ratio)
    test_data_size = int(len(data_array) * test_ratio)
    train_data = data_array[:train_data_size]
    test_data = data_array[train_data_size:train_data_size + test_data_size]
    
    feature_train = train_data[:, :-1]  # All columns except the last one
    label_train = train_data[:, -1]     # Last column as labels
    label_train = label_train.astype(int)  # Ensure labels are integers
    feature_test = test_data[:, :-1]      # All columns except the last one
    label_test = test_data[:, -1]        # Last column as labels
    label_test = label_test.astype(int)    # Ensure labels are integers
    return feature_train, label_train, feature_test, label_test


data = load_data(DatasetDirectory.HEART_DISEASE)
data_preprocessing(data, DatasetDirectory.HEART_DISEASE)
feature_train, label_train, feature_test, label_test = prepare_dataset(data, 0.8, 0.2)

clf = DecisionTreeClassifier(criterion="entropy", random_state=42, max_depth=4)

clf.fit(feature_train, label_train)

label_pred = clf.predict(feature_test)
print("Predicted labels:", label_pred)
print("Actual labels:", label_test)