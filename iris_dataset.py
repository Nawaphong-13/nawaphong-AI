import pandas as pd
import numpy as np 
import os

def load(path='./dataset/iris.csv', split_train_test=None):
    if os.path.isfile(path):
        iris = pd.read_csv(path)
   
    else:
        url = 'http://archive.ics.uci.edu/ml/machine-learning-datasets/iris/iris.data'
        iris = pd.read_csv(url, header=None)
        iris.to_csv(path, index=False)
    X = iris.iloc[:,:4].values
    Y = iris.iloc[:,-1].values
    if split_train_test:
        classes = np.unique(Y)
        itrain = np.empty((0,), dtype=np.int)
        itest = np.empty((0,), dtype=np.int)
        for i in classes:
            idx = np.where(Y==i)[0]
            split = int(len(idx) * split_train_test)
            itrain = np.concatenate((itrain, idx[:split]))
            itest = np.concatenate((itest, idx[split:]))
        return X[itrain], Y[itrain], X[itest], Y[itest]
    return X, Y

if __name__ == "__main__":
    Xtrain, Ytrain, Xtest, Ytest = load(split_train_test=0.5)
    print(Xtrain.shape)
    print(Ytrain.shape)