from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import time
import random

from ..model.model import Model
from .model_init import init_model

from ..embedding.tf_idf import TFIDF_Wrapper
from ..embedding.word2vec import W2V_Wrapper
from ..embedding.onehot import OneHot

from ..data.hyperparams import *

def shuffle_text_and_label(texts, label):
    combined_data = list(zip(texts, label))
    # Shuffle the combined data
    random.shuffle(combined_data)
    # Unzip the shuffled data back into separate lists
    shuffled_texts, shuffled_label = zip(*combined_data)

    return shuffled_texts, shuffled_label


class KFoldTest:
    def __init__(self, embedding_type, classify_type, n_splits=5, random_state=42, shuffle=True):
        assert embedding_type in ["tf-idf", "word2vec", "one-hot", "finetune"]
        self.embedding_type = embedding_type
        self.classify_type = classify_type
        self.kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    def run(self, data):
        accuracies = []
        f1_scores = []
        train_times = []
        prediction_times = []

        labels = [item["label"] for item in data]
        texts = [item["raw"] for item in data]
        
        for train_idx, dev_idx in self.kf.split(data):
            # Split data into training and development sets
            train_data = [data[i] for i in train_idx]
            dev_data = [data[i] for i in dev_idx]

            train_label = [labels[idx] for idx in train_idx]
            dev_label = [labels[idx] for idx in dev_idx]

            train_texts = [item["raw"] for item in train_data]
            dev_texts = [item["raw"] for item in dev_data]
            tokenized_texts = [text.split() for text in train_texts]
            
            train_texts, train_label = shuffle_text_and_label(train_texts, train_label)
            dev_texts, dev_label = shuffle_text_and_label(dev_texts, dev_label)

            if self.embedding_type == "word2vec":
                word2vec_hp = Word2VecHP()
                w2v = W2V_Wrapper(input_sentence=tokenized_texts, params=word2vec_hp)
                # Convert text data to word vectors
                train_vectors = np.array([w2v.text_to_vector_padding(item, MAX_SEQ_LENGTH) for item in train_texts])
                dev_vectors = np.array([w2v.text_to_vector_padding(item, MAX_SEQ_LENGTH) for item in dev_texts])
                # train_vectors = [w2v.text_to_vector(item["raw"], mean=False) for item in train_data]
                # dev_vectors = [w2v.text_to_vector(item["raw"], mean=False) for item in dev_data]
                # print(train_vectors[0])
            elif self.embedding_type == "tf-idf":
                tf_idf_hp = TFIDF_HP()
                tf_idf = TFIDF_Wrapper(corpus=tokenized_texts, params=tf_idf_hp)
                # Convert text data to frequency vectors
                train_vectors = tf_idf.doc_to_vector(train_texts)
                dev_vectors = tf_idf.doc_to_vector(dev_texts)
                # print(train_vectors)
            elif self.embedding_type == "one-hot":
                one_hot_hp = OneHotHP()
                one_hot = OneHot(train_texts, one_hot_hp)
                train_vectors = one_hot.corpus_to_one_hot(train_texts)
                dev_vectors = one_hot.corpus_to_one_hot(dev_texts)
                # print(train_vectors[0])

            ############# Pre-Training methods encoding ##############
            else:
                train_vectors = train_texts
                dev_vectors = dev_texts
                print("Embedding in Pretrained model, not separately.")

            # print("Embedding Finished..")

            # Train the model
            self.classify_model = init_model(self.classify_type)
            start_time = time.time()
            self.classify_model.train(train_vectors, train_label)
            end_time = time.time()
            train_time = end_time - start_time
            train_times.append(train_time)

            # Make predictions on the development set
            start_time = time.time()
            dev_predictions = self.classify_model.predict(dev_vectors)
            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)
            # print("Prediction: ", dev_predictions)

            # Calculate accuracy and store it
            accuracy = accuracy_score(dev_label, dev_predictions)
            f1 = f1_score(dev_label, dev_predictions, average='weighted')
            accuracies.append(accuracy)
            f1_scores.append(f1)

        # Logging out time test result    
        mean_train_time = np.mean(train_times)
        mean_prediction_time = np.mean(prediction_times)
        print(f"Mean Training Time: {mean_train_time:.2f} seconds")
        print(f"Mean Prediction Time: {mean_prediction_time:.2f} seconds")

        # Calculate and print the mean accuracy across folds
        mean_accuracy = np.mean(accuracies)
        print(f"Mean Accuracy: {mean_accuracy:.4f}")
        mean_f1 = np.mean(f1_scores)
        print(f"Mean F1 score: {mean_f1:.4f}")
