import sys
import os

######### Validation Settings #########
# Setting true to run validations, 
# for BERT and XLNet, open your VPN, make sure you running on GPU with 24G memory or more
validation = {
    "Random": True,
    "NaiveBayes": True,
    "Softmax": True,
    "DecisionTree": True,
    "SVM": True,
    "MLP": True,
    "CNN": True,
    "RNN": True,
    "BERT": False,
    "XLNet": False,
}

# Get the parent directory of the current script (i.e., the root directory of your project)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the project root to the Python path
sys.path.append(project_root)

from dmac.io.loader import Project1Loader

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

if validation["Random"]:
    print("Running Random Model")
    random_kftest = KFoldTest(embedding_type='tf-idf', classify_type="Random")
    random_kftest.run(data)

if validation["SVM"]:
    print("Running SVM Model")
    svm_kftest = KFoldTest(embedding_type="tf-idf", classify_type="SVM")
    svm_kftest.run(data)

if validation["Softmax"]:
    print("Running Softmax Model")
    softmax_kftest = KFoldTest(embedding_type="tf-idf", classify_type="Softmax")
    softmax_kftest.run(data)

if validation["DecisionTree"]:
    print("Running DecisionTree Model")
    dt_kftest = KFoldTest(embedding_type="tf-idf", classify_type="DecisionTree")
    dt_kftest.run(data)

if validation["NaiveBayes"]:
    print("Running NaiveBayes Model")
    dt_kftest = KFoldTest(embedding_type="tf-idf", classify_type="NaiveBayes")
    dt_kftest.run(data)

if validation["MLP"]:
    print("Running MLP Model")
    mlp_kftest = KFoldTest(embedding_type="tf-idf", classify_type="MLP")
    mlp_kftest.run(data)

if validation["RNN"]:
    print("Running RNN Model")
    rnn_kftest = KFoldTest(embedding_type="word2vec", classify_type="RNN")
    rnn_kftest.run(data)

if validation["CNN"]:
    print("Running CNN Model")
    cnn_kftest = KFoldTest(embedding_type="word2vec", classify_type="CNN")
    cnn_kftest.run(data)

if validation["BERT"]:
    print("Running BERT Model")
    bert_kftest = KFoldTest(embedding_type='finetune', classify_type="BERT")
    bert_kftest.run(data)

if validation["XLNet"]:
    print("Running XLNet Model")
    xlnet_kftest = KFoldTest(embedding_type='finetune', classify_type="XLNet")
    xlnet_kftest.run(data)



