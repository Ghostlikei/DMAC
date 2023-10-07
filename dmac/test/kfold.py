from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np

from ..model.model import Model

class KFoldTest:
    def __init__(self, embedding_model, classify_model: Model, n_splits=5, random_state=42, shuffle=True):
        self.embedding_model = embedding_model
        self.classify_model = classify_model
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def run(self, data):
        accuracies = []
        f1_scores = []

        labels = [item["label"] for item in data]
        texts = [item["raw"] for item in data]
        
        for train_idx, dev_idx in self.kf.split(data):
            # Split data into training and development sets
            train_data = [data[i] for i in train_idx]
            dev_data = [data[i] for i in dev_idx]

            train_label = [labels[idx] for idx in train_idx]
            # print(train_data[0])
            dev_label = [labels[idx] for idx in dev_idx]

            # Convert text data to word vectors
            train_vectors = np.array([self.embedding_model.text_to_vector(item["raw"]) for item in train_data])
            # print(train_vectors[0])
            dev_vectors = np.array([self.embedding_model.text_to_vector(item["raw"]) for item in dev_data])

            assert len(train_label) == len(train_vectors)
            assert len(dev_label) == len(dev_vectors)

            # Train the model
            self.classify_model.train(train_vectors, train_label)

            # Make predictions on the development set
            dev_predictions = self.classify_model.predict(dev_vectors)
            print("Prediction: ", dev_predictions)

            # Calculate accuracy and store it
            accuracy = accuracy_score(dev_label, dev_predictions)
            f1 = f1_score(dev_label, dev_predictions, average='weighted')
            accuracies.append(accuracy)
            f1_scores.append(f1)

        # Calculate and print the mean accuracy across folds
        mean_accuracy = np.mean(accuracies)
        print(f"Mean Accuracy: {mean_accuracy:.2f}")
        mean_f1 = np.mean(f1_scores)
        print(f"Mean F1 score: {mean_f1:.2f}")