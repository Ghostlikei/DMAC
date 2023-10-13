from ..data.hyperparams import *

from ..model.model import Model

from ..model.random import Random

from ..model.svm import SVM
from ..model.softmax import Softmax
from ..model.decisiontree import DecisionTree
from ..model.naivebayes import NaiveBayes

from ..model.mlp import MLP
from ..model.rnn import RNN
from ..model.cnn import CNN

from ..model.bert import BERT
from ..model.xlnet import XLNet

def init_model(type):
    if type == "Random":
        return Random()

    elif type == "SVM":
        return SVM(SVM_RBF_HP())

    elif type == "Softmax":
        return Softmax(SoftmaxHP())
    
    elif type == "DecisionTree":
        return DecisionTree(DecisionTreeHP())

    elif type == "NaiveBayes":
        return NaiveBayes(NaiveBayesHP())

    elif type == "MLP":
        return MLP(MLP_HP())

    elif type == "CNN":
        return CNN(CNN_HP())

    elif type == "RNN":
        return RNN(RNN_HP())

    elif type == "BERT":
        return BERT(BertHP())
    
    elif type == "XLNet":
        return XLNet(XLNetHP())

    else:
        raise NotImplementedError

