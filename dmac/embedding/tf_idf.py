from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TFIDF_Wrapper:
    def __init__(self, corpus, params):
        preprocessed_texts = [' '.join(tokens) for tokens in corpus]

        # Step 2: Create a TF-IDF Vectorizer and fit it on the entire corpus
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=params.max_features,
            stop_words=params.stop_words,
        )
        self.tfidf_vectorizer.fit(preprocessed_texts)

    def doc_to_vector(self, doc):
        return self.tfidf_vectorizer.transform(doc)
