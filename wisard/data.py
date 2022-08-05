from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


class IrisDataset:

    def __init__(self, test_size: int = 0.2):
        self.test_size = test_size

    def load_data(self):
        X, y = load_iris(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y)
        return (X_train, y_train), (X_test, y_test)