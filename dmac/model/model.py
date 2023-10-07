from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def train(self, train_data, train_label):
        return NotImplemented

    @abstractmethod
    def predict(self, predict_data):
        return NotImplemented