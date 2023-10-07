import sys
import os

# Get the parent directory of the current script (i.e., the root directory of your project)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Append the project root to the Python path
sys.path.append(project_root)

from dmac.io.loader import Project1Loader
from dmac.embedding.word2vec import W2V_Wrapper
from dmac.model.svm import SVM
from dmac.data.hyperparams import *
from dmac.test.kfold import KFoldTest

path = "../data/exp1_data/train_data.txt"

ld = Project1Loader()
data = ld.load(path)

labels = [item["label"] for item in data]
texts = [item["raw"] for item in data]

tokenized_texts = [text.split() for text in texts]
print(len(tokenized_texts[0]))

word2vec_hp = Word2VecHP()
embedding_model = W2V_Wrapper(input_sentence=tokenized_texts, params=word2vec_hp)

svm_rbf_hp = SVM_RBF_HP()
svm_rbf = SVM(svm_rbf_hp)

svm_kftest = KFoldTest(embedding_model=embedding_model, classify_model=svm_rbf)
svm_kftest.run(data)