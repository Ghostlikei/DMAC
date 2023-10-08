"""
Designed for testing whether other models take effect
"""
import numpy as np
from .model import Model

class Random(Model):
    def __init__(self, params=None):
        pass

    def train(self, train_data, train_label):
        # Random classifier doesn't require training, 
        # So we'll just store the unique labels from the training data.
        self.unique_labels = np.unique(train_label)

    def predict(self, predict_data):
        # Generate random predictions by randomly selecting labels from the unique labels seen during training.
        num_samples = predict_data.shape[0]
        random_predictions = np.random.choice(self.unique_labels, size=num_samples)
        return random_predictions