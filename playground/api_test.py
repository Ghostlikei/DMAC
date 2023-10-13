import sys
import os

# Get the parent directory of the current script (i.e., the root directory of your project)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the project root to the Python path
sys.path.append(project_root)

from dmac.io.loader import Project1Loader

# Embeddings
from dmac.embedding.word2vec import W2V_Wrapper
from dmac.embedding.tf_idf import TFIDF_Wrapper

# Models
from dmac.model.random import Random

from dmac.model.svm import SVM
from dmac.model.softmax import Softmax
from dmac.model.decisiontree import DecisionTree
from dmac.model.naivebayes import NaiveBayes

from dmac.model.mlp import MLP
from dmac.model.rnn import RNN
from dmac.model.cnn import CNN

from dmac.model.bert import BERT
from dmac.model.xlnet import XLNet

# Hyperparams
from dmac.data.hyperparams import *

# Tests
from dmac.test.kfold import KFoldTest

path = "../data/exp1_data/train_data.txt"

ld = Project1Loader()
data = ld.load(path)

# labels = [item["label"] for item in data]
# texts = [item["raw"] for item in data]

# tokenized_texts = [text.split() for text in texts]
# print(len(tokenized_texts[0]))

# random_clf = Random()
# random_kftest = KFoldTest(embedding_type='tf-idf', classify_model=random_clf)
# random_kftest.run(data)

# svm_rbf_hp = SVM_RBF_HP()
# svm_rbf = SVM(svm_rbf_hp)
# svm_kftest = KFoldTest(embedding_type="tf-idf", classify_model=svm_rbf)
# svm_kftest.run(data)

# softmax_hp = SoftmaxHP()
# softmax = Softmax(softmax_hp)
# softmax_kftest = KFoldTest(embedding_type="one-hot", classify_model=softmax)
# softmax_kftest.run(data)

# dt_hp = DecisionTreeHP()
# decision_tree = DecisionTree(dt_hp)
# dt_kftest = KFoldTest(embedding_type="one-hot", classify_model=decision_tree)
# dt_kftest.run(data)

# nb_hp = NaiveBayesHP()
# naive_bayes = NaiveBayes(nb_hp)
# dt_kftest = KFoldTest(embedding_type="tf-idf", classify_model=naive_bayes)
# dt_kftest.run(data)

mlp_hp = MLP_HP()
mlp = MLP(mlp_hp)
mlp_kftest = KFoldTest(embedding_type="tf-idf", classify_model=mlp)
mlp_kftest.run(data)

#rnn = RNN(RNN_HP())
#rnn_kftest = KFoldTest(embedding_type="word2vec", classify_model=rnn)
#rnn_kftest.run(data)

#cnn = CNN(CNN_HP())
#cnn_kftest = KFoldTest(embedding_type="word2vec", classify_model=cnn)
#cnn_kftest.run(data)
        
# bert_clf = BERT(BertHP())
# bert_kftest = KFoldTest(embedding_type='finetune', classify_model=bert_clf)
# bert_kftest.run(data)

# xlnet_clf = XLNet(XLNetHP())
# xlnet_kftest = KFoldTest(embedding_type='finetune', classify_model=xlnet_clf)
# xlnet_kftest.run(data)

