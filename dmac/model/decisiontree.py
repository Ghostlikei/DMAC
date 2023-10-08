from .model import Model
from sklearn.tree import DecisionTreeClassifier

class DecisionTree(Model):
    def __init__(self, params):
        self.clf = DecisionTreeClassifier(
            criterion=params.criterion,
            max_depth=params.max_depth,
            min_samples_split=params.min_samples_split,
            min_samples_leaf=params.min_samples_leaf,
            random_state=params.random_state
        )

    def train(self, train_data, train_label):
        self.clf.fit(train_data, train_label)

    def predict(self, predict_data):
        return self.clf.predict(predict_data)
