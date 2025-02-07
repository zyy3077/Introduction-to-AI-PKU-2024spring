import numpy as np
from PIL import Image, ImageFilter
trn_X = np.load('MNIST/train_data.npy').astype(np.float64)
trn_Y = np.load('MNIST/train_targets.npy')
trn_num_sample = trn_X.shape[0]
trn_X = trn_X.reshape(trn_num_sample, -1)
std_X, mean_X = np.std(trn_X, axis=0, keepdims=True)+1e-4, np.mean(trn_X, axis=0, keepdims=True)
num_feat = trn_X.shape[1]
num_class = np.max(trn_Y) + 1

val_X = np.load('MNIST/valid_data.npy').astype(np.float64)
val_Y = np.load('MNIST/valid_targets.npy')
val_X = val_X.reshape(val_X.shape[0], -1)

test_X = np.load('MNIST/test_data.npy').astype(np.float64)
test_Y = np.load('MNIST/test_targets.npy')

num_data = test_X.shape[0]

def getdata(idx):
    return test_X[idx]

def gety(idx):
    return test_Y[idx]

def getdatasets(y):
    return np.arange(num_data)[test_Y==y]
