import numpy as np
from collections import Counter

class OneHot:
    def __init__(self, corpus, params):
        self.corpus = corpus
        self.n_frequent = params.n_frequent
        self.vocab = self.build_vocab(corpus)

    def build_vocab(self, build_corpus):
        # Create a vocabulary of the top n_frequent words from the corpus
        word_counts = Counter()
        for sentence in build_corpus:
            words = sentence.split()
            word_counts.update(words)
        # Select the top n_frequent words
        most_common_words = word_counts.most_common(self.n_frequent)
        vocab = {word: idx for idx, (word, _) in enumerate(most_common_words)}
        # Add an "out of vocabulary" category
        vocab['<OOV>'] = len(vocab)
        return vocab

    def sentence_to_one_hot(self, sentence):
        # Convert a sentence to a one-hot vector based on the vocabulary
        one_hot_vector = np.zeros(len(self.vocab), dtype=int)
        words = sentence.split()
        for word in words:
            if word in self.vocab:
                word_index = self.vocab[word]
                one_hot_vector[word_index] = 1
            else:
                # Out of vocabulary word
                one_hot_vector[-1] = 1
        return one_hot_vector

    def corpus_to_one_hot(self, input_corpus):
        # Convert the entire corpus to one-hot vectors
        one_hot_vectors = [self.sentence_to_one_hot(sentence) for sentence in input_corpus]
        return np.array(one_hot_vectors)
