from .model import Model
from sklearn.linear_model import LogisticRegression

class Softmax(Model):
    def __init__(self, params):
        self.clf = LogisticRegression(
            solver='lbfgs',  # You can choose a different solver if needed
            multi_class='multinomial',
            C=params.C
        )

    def train(self, train_data, train_label):
        self.clf.fit(train_data, train_label)

    def predict(self, predict_data):
        return self.clf.predict(predict_data)
