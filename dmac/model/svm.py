from .model import Model
from sklearn.svm import SVC

class SVM(Model):
    def __init__(self, params):
        self.clf = SVC(kernel=params.kernel,
                       C=params.C,
                       gamma=params.gamma)

    def train(self, train_data, train_label):
        self.clf.fit(train_data, train_label)

    def predict(self, predict_data):
        return self.clf.predict(predict_data)