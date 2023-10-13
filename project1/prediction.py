import numpy as np
import time
import random

import sys
import os

# Get the parent directory of the current script (i.e., the root directory of your project)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the project root to the Python path
sys.path.append(project_root)

from dmac.model.model import Model
from dmac.test.model_init import init_model
from dmac.model.bert import BERT

from dmac.io.loader import Project1Loader

from dmac.embedding.tf_idf import TFIDF_Wrapper
from dmac.embedding.word2vec import W2V_Wrapper
from dmac.embedding.onehot import OneHot

from dmac.data.hyperparams import *

def shuffle_text_and_label(texts, label):
    combined_data = list(zip(texts, label))
    # Shuffle the combined data
    random.shuffle(combined_data)
    # Unzip the shuffled data back into separate lists
    shuffled_texts, shuffled_label = zip(*combined_data)

    return shuffled_texts, shuffled_label

def run():
    path = "../data/exp1_data/train_data.txt"
    classify_type = "BERT"

    ld = Project1Loader()
    data = ld.load(path)

    test_sentences = []

    # Specify the path to your 'test.txt' file
    file_path = '../data/exp1_data/test.txt'

    # Open and read the file line by line
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Remove the initial "0, " part and append the sentence to the list
            sentence = line.split(', ', 1)[-1].strip()
            test_sentences.append(sentence)

    test_sentences = test_sentences[1:]

    # Now, 'sentences' contains the sentences without the initial "0, " part
    print(test_sentences[0:2])

    labels = [item["label"] for item in data]
    texts = [item["raw"] for item in data]

    texts, labels = shuffle_text_and_label(texts, labels)

    classify_model = init_model(classify_type)
    classify_model.train(texts, labels)

    test_labels = classify_model.predict(test_sentences)

    output_file = '../data/exp1_data/output_labels.txt'

    # Create and write the id, pred pairs to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('id, pred\n')
        for i in range(len(test_labels)):
            file.write(f'{i}, {test_labels[i]}\n')


if __name__ == '__main__':
    run()
