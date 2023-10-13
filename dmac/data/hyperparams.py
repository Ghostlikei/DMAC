"""
File: hyperparams.py
Designed to save all the hyperparams of dmac module, all in one file
"""
########### Misc Hyperparams #################
MAX_SEQ_LENGTH = 200 # Used for word2vec padding of non-fixed length of sentences, a common method in NLP
EMBEDDING_SIZE = 100

########### Embedding Model Hyperparams #############
class TFIDF_HP:
    def __init__(self):
        self.max_features = 5000  # Maximum number of features (terms) to keep
        self.stop_words = 'english'  # Stop words to be removed ('english' removes common English stop words)
        self.tokenizer = None  # You can define your custom tokenizer function here if needed

class Word2VecHP:
    def __init__(self):
        # Hyperparameters for Word2Vec
        self.vector_size = EMBEDDING_SIZE
        self.glove_type = "glove-twitter-100"

class OneHotHP:
    def __init__(self):
        self.n_frequent = 5000

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

### Classification Model With Deep Learning Hyperparams #########
class MLP_HP:
    def __init__(self):
        # Hyperparameters for the MLP model
        self.input_size = 5000
        self.hidden_size = 64  # Number of neurons in the hidden layer
        self.output_size = 10  # Number of output classes (0 to 9)
        self.learning_rate = 0.01  # Learning rate for optimization
        self.batch_size = 32  # Mini-batch size for training
        self.num_epochs = 300  # Number of training epochs

        self.dropout_prob = 0.5

class RNN_HP:
    def __init__(self):
        # RNN Hyperparameters
        self.type = "GRU"
        self.bidirectional = True

        self.input_size = EMBEDDING_SIZE  # Size of input features (e.g., embedding size)
        self.hidden_size = 128  # Size of the hidden state in the RNN
        self.num_layers = 2    # Number of RNN layers
        self.output_size = 10  # Number of output classes

        self.batch_size = 32   # Mini-batch size for training
        self.learning_rate = 0.001  # Learning rate for the optimizer
        self.weight_decay = 1e-4

        self.num_epochs = 15   # Number of training epochs

        self.dropout_prob = 0.3

class CNN_HP:
    def __init__(self):
        # Setting CNN hyperparameters
        self.embedding_dim = EMBEDDING_SIZE  # Dimension of word embeddings
        self.num_filters = 64  # Number of filters in the convolutional layers
        self.filter_sizes = [3, 4, 5]  # List of filter sizes for convolution

        self.output_size = 10  # Number of classes for classification

        self.learning_rate = 0.001  # Learning rate for optimization
        
        self.batch_size = 32  # Batch size for training

        self.num_epochs = 70  # Number of training epochs

        self.dropout_prob = 0.2

class BertHP:
    def __init__(self):
        self.bert_model_name = "bert-base-uncased"  # Pre-trained BERT model name
        self.num_labels = 10  # Number of classification labels
        self.learning_rate = 1e-5  # Learning rate for optimization
        
        self.batch_size = 16  # Mini-batch size for training
        self.num_epochs = 5  # Number of training epochs
        self.max_seq_length = MAX_SEQ_LENGTH  # Maximum sequence length for BERT input

class XLNetHP:
    def __init__(self):
        self.xlnet_model_name = "xlnet-base-cased"  # Pre-trained XLNet model name
        self.num_labels = 10  # Number of classification labels
        self.learning_rate = 2e-5  # Learning rate for optimization
        self.batch_size = 1  # Mini-batch size for training
        self.num_epochs = 3  # Number of training epochs
        self.max_seq_length = MAX_SEQ_LENGTH  # Maximum sequence length for XLNet input

