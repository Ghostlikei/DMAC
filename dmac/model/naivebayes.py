from .model import Model
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(Model):
    def __init__(self, params):
        self.clf = MultinomialNB(alpha=params.alpha)

    def train(self, train_data, train_label):
        self.clf.fit(train_data, train_label)

    def predict(self, predict_data):
        return self.clf.predict(predict_data)