"""
File: hyperparams.py
Designed to save all the hyperparams of dmac module, all in one file
"""

########### Embedding Model Hyperparams #############
class TFIDF_HP:
    def __init__(self):
        self.max_features = 5000  # Maximum number of features (terms) to keep
        self.stop_words = 'english'  # Stop words to be removed ('english' removes common English stop words)
        self.tokenizer = None  # You can define your custom tokenizer function here if needed

class Word2VecHP:
    def __init__(self):
        # Hyperparameters for Word2Vec
        self.vector_size = 100
        self.window = 5
        self.min_count = 1
        self.sg = 0  # 0 for CBOW, 1 for Skip-gram

########### Misc Hyperparams #################
MAX_SEQ_LENGTH = 200 # Used for word2vec padding of non-fixed length of sentences, a common method in NLP

### Classification Model Hyperparams #########
class SVM_RBF_HP:
    def __init__(self):
        self.kernel = "rbf"
        self.C = 1.0
        self.gamma = "scale"

class SoftmaxHP:
    def __init__(self):
        self.C = 1.0

class DecisionTreeHP:
    def __init__(self):
        """
        criterion (str, optional): The function to measure the quality of a split. Default is 'gini'.
        max_depth (int, optional): The maximum depth of the tree. Default is None.
        min_samples_split (int or float, optional): The minimum number of samples required to split an internal node.
            Default is 2.
        min_samples_leaf (int or float, optional): The minimum number of samples required to be at a leaf node.
            Default is 1.
        random_state (int or None, optional): The random seed for reproducibility. Default is None.
        """
        self.criterion = 'entropy'
        self.max_depth = None
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.random_state = None

class NaiveBayesHP:
    def __init__(self):
        """
        Parameters:
            alpha (float): The smoothing parameter (Laplace smoothing). Default is 1.0.
        """
        self.alpha = 1.0

