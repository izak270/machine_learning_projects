from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Training:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)
        self.train_test = [x_train, x_test, y_train, y_test]

    def train(self):
        self.model.fit(self.train_test[0], self.train_test[1])
