from gensim.models import Word2Vec
from gensim import downloader
import numpy as np
import re

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Split text into words and remove non-alphanumeric characters
    words = re.findall(r'\b\w+\b', text)

    # Join the words back into a single string
    processed_text = ' '.join(words)

    return processed_text

class W2V_Wrapper:
    def __init__(self, input_sentence, params):
        """
        Using Glove as embedding method
        """
        self.vector_size = params.vector_size
        self.embedding_model = downloader.load(params.glove_type)

    def text_to_vector(self, text: str, mean=True):
        """
        `text`: A string of text to be embedded. \n
        `mean`: Define whether returns the full token embedding(tensor) or mean embedding(vector)
        """
        text = preprocess_text(text)
        words = text.split()
        # Aggregate word vectors (skip words not in the vocabulary)
        word_vectors = [self.embedding_model[word] for word in words if word in self.embedding_model]
        if len(word_vectors) > 0:
            if mean:
                # Why?
                return np.mean(word_vectors, axis=0)
            else:
                # This case is not applicable, because it's hard to do ML classification in tensor space R^(n*m)
                # So we need to conpress it into linear space R^n (mean method above)
                # Why not just concate one vector after another word? Like [[0.1, 0.2], [0.3, 0.4]]->[0.1, 0.2, 0.3, 0.4]?
                # If you do so, Why not use one-hot vector? That saves the embedding size right?
                # So for Classification Algorithms works on Linear space R^n, just use one-hot vector
                # You can also use word2vec and do mapping from tensor space into linear space, but it works not good.
                return word_vectors
        else:
            return np.zeros(self.embedding_model.vector_size)

    # def text_to_vector_padding(self, text: str, max_seq_length):
    #     words = text.split()
    #     # Truncate or pad the sequence to the specified max_seq_length
    #     if len(words) < max_seq_length:
    #         words = words + ['<PAD>'] * (max_seq_length - len(words))
    #     else:
    #         words = words[:max_seq_length]

    #     # Create a mask to indicate real words (1 for real words, 0 for padding)
    #     # Is that necessary?
    #     # mask = [1] * len(words) + [0] * (max_seq_length - len(words))

    #     # word_vectors = [self.embedding_model.wv[word] for word in words if word in self.embedding_model.wv]
    #     word_vectors = [self.embedding_model.wv[word] for word in words]
    #     print(len(word_vectors))

    #     if len(word_vectors) > 0:
    #         # return word_vectors, np.array(mask)
    #         return word_vectors
    #     else:
    #         return np.zeros((max_seq_length * self.embedding_model.vector_size)), np.zeros(max_seq_length)

    def text_to_vector_padding(self, text: str, max_seq_length):
        text = preprocess_text(text)
        words = text.split()
        
        # Initialize an empty list to store word vectors
        word_vectors = []
        cnt0 = 0
        cnt1 = 0
        
        for word in words:
            if word in self.embedding_model:
                cnt0 += 1
                # If the word exists in the embedding model, use its vector
                word_vectors.append(self.embedding_model[word])
            else:
                cnt1 += 1
                # If the word is not in the embedding model, use an all-zero vector
                # zero_vector = list(np.random.normal((-1, 1), self.vector_size))
                zero_vector = [0.0] * self.vector_size
                word_vectors.append(zero_vector)
        
        # Truncate or pad the sequence to the specified max_seq_length
        if len(word_vectors) < max_seq_length:
            word_vectors += [[0.0] * self.vector_size] * (max_seq_length - len(word_vectors))
        else:
            word_vectors = word_vectors[:max_seq_length]

        # print(f"Word in dict: {cnt0}, not in dict: {cnt1}")

        return word_vectors


## Maybe designing a GloVe wrapper in the future