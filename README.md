# DMAC
Recording personal code design for DaSE Modern AI course

Github repo:https://github.com/Ghostlikei/DMAC Codes maybe updated later to testing LLM apis

## Project1

Validation code locates on `/DMAC/project1/validation.py`，

Prediction of testing set locates on `/DMAC/data/exp1_data/results.txt`

My report locates on `/DMAC/report`, with 2 version of `.pdf` and `.md`

Requirements are saved on `/DMAC/dmac/requirements.txt`. However, it is autogenerated, missing some internal including packages. So if you have problem running validation code, please follow the installation guide below. If you are still not able to run the validation code, please turn on your VPN, follow the import error message or contact me through e-mail or issue.

Modular testing architecture is located on `/DMAC/dmac`, it contains 5 parts:

- `io`

Loader and Writer(not implemented) of datasets

- `model`

Text Classification model. All models are derived from abstract class `model/model.py`,which must contains `train()`method and `predict()`method

- `test`

Contains k-fold testing codes. 

- `embedding`

Contains tf-idf, one-hot and word2vec embedding APIs

- `data`

All hyperparams are save in `hyperparams.py`, If you wanna build autotuner, just build your own hyperparams class. In my code, hyperparams are hard-coded

Also some helpful samples and irrunnable codes in `/DMAC/playground/`

---

Default testing models: Random，SVM，DecisionTree，NaiveBayes，Softmax, MLP, CNN, RNN

If you wanna test CNN/RNN, it's better running the code on GPU,

If you wanna test BERT/XLNet, you should set proxy to huggingface.io, and make sure the GPU Memory is at least 24G

Running installation for python ThirdParty package downloading:

```sh
pip install transformers
pip install gensim
pip install scikit-learn
pip install accelerate -U
pip install sentencepiece
```

If your environment does not contain basic ML/DL models, you should install them manually according to the error message(Including package like scikit-learn, pytorch, numpy and so on).

Any question about the simple testing architecture, please raise **ISSUE** on github directly.

Due to the limitation of time, the `dmac` testing architecture would maybe too simple and buggy. If you wanna modify the testing architecture, please directly fork this repo.

## Project 2

A* Algorithm and its modification

All code in `project2`