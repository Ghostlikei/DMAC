from gensim.models import Word2Vec
import numpy as np


class W2V_Wrapper:
    def __init__(self, input_sentence, params):
        """
        Using CBOW as embedding method
        """
        self.embedding_model = Word2Vec(sentences=input_sentence,
                                        vector_size=params.vector_size,
                                        window=params.window,
                                        min_count=params.min_count,
                                        sg=params.sg)

    def text_to_vector(self, text: str, mean=True):
        """
        `text`: A string of text to be embedded. \n
        `mean`: Define whether returns the full token embedding(tensor) or mean embedding(vector)
        """
        words = text.split()
        # Aggregate word vectors (skip words not in the vocabulary)
        word_vectors = [self.embedding_model.wv[word] for word in words if word in self.embedding_model.wv]
        if len(word_vectors) > 0:
            if mean:
                return np.mean(word_vectors, axis=0)
            else:
                return word_vectors
        else:
            return np.zeros(self.embedding_model.vector_size)

## Maybe designing a GloVe wrapper in the future