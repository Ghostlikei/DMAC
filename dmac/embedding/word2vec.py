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

    def text_to_vector_padding(self, text: str, max_seq_length):
        words = text.split()
        # Truncate or pad the sequence to the specified max_seq_length
        if len(words) < max_seq_length:
            words = words + ['<PAD>'] * (max_seq_length - len(words))
        else:
            words = words[:max_seq_length]

        # Create a mask to indicate real words (1 for real words, 0 for padding)
        # Is that necessary?
        # mask = [1] * len(words) + [0] * (max_seq_length - len(words))

        word_vectors = [self.embedding_model.wv[word] for word in words if word in self.embedding_model.wv]

        if len(word_vectors) > 0:
            # return word_vectors, np.array(mask)
            return word_vectors
        else:
            return np.zeros((max_seq_length * self.embedding_model.vector_size)), np.zeros(max_seq_length)

## Maybe designing a GloVe wrapper in the future