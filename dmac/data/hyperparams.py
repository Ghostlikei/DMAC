"""
File: hyperparams.py
Designed to save all the hyperparams of dmac module, all in one file
"""

class Word2VecHP:
    def __init__(self):
        # Hyperparameters for Word2Vec
        self.vector_size = 100
        self.window = 5
        self.min_count = 1
        self.sg = 0  # 0 for CBOW, 1 for Skip-gram

class SVM_RBF_HP:
    def __init__(self):
        self.kernel = "rbf"
        self.C = 1.0
        self.gamma = "scale"

class SoftmaxHP:
    def __init__(self):
        self.C = 1.0  # You can adjust the regularization parameter C as needed
