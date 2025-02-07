import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed
from copy import deepcopy
from typing import List
from autograd.BaseGraph import Graph
from autograd.utils import buildgraph
from autograd.BaseNode import *
from scipy.ndimage import rotate, shift, zoom, center_of_mass
setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/your.npy"
# X = mnist.trn_X
# Y = mnist.trn_Y
X = np.concatenate([mnist.trn_X, mnist.val_X], axis=0)
Y = np.concatenate([mnist.trn_Y, mnist.val_Y], axis=0)

lr = 1e-3
wd1 = 0
wd2 = 1e-5
batchsize = 128
p = 0.0

def augment_data(dataset, label):
    augmented_data = []
    augmented_label = []
    for i, img in enumerate(dataset):
        img = img.reshape(28, 28)

        rotated = rotate(img, np.random.uniform(-15, 15), reshape=False)
        augmented_data.append(rotated.reshape(-1))
        augmented_label.append(label[i])

        translated = shift(img, [np.random.randint(-5, 5), np.random.randint(-5, 5)])
        augmented_data.append(translated.reshape(-1))
        augmented_label.append(label[i])

        # zoom_factor = np.random.uniform(0.8, 1.2)
        # zoomed = zoom(img, zoom_factor)
        # if zoom_factor < 1:
        #     zoomed = np.pad(zoomed, ((0, 28 - zoomed.shape[0]), (0, 28 - zoomed.shape[1])), 'constant')
        # else:
        #     y, x = center_of_mass(img)
        #     x0 = int(x - 14)
        #     y0 = int(y - 14)
        #     zoomed = img[y0 : y0 + 28, x0 : x0 + 28]
        # augmented_data.append(zoomed.reshape(-1))
        # augmented_label.append(label[i])

    return (np.array(augmented_data), np.array(augmented_label))

_X, _Y = augment_data(X, Y)
X = np.concatenate([X, _X], axis=0)
Y = np.concatenate([Y, _Y], axis=0)

if __name__ == "__main__":
    
    # graph = Graph([Linear(784, 256), BatchNorm(256), relu(), Dropout_Corrected(p), 
    #                Linear(256, 128), BatchNorm(128), relu(),  
    #                Linear(128, 64), BatchNorm(64), relu(),  
    #                Linear(64, 32), BatchNorm(32), relu(),  
    #                Linear(32, 10), LogSoftmax(), NLLLoss(Y)])
    graph = Graph([Linear(784, 256), BatchNorm(256), relu(), Dropout_Corrected(p), 
                   Linear(256, 64), BatchNorm(64), relu(),   
                   Linear(64, 10), LogSoftmax(), NLLLoss(Y)])
    
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)
    for i in range(1, 35+1):
        hatys = []
        ys = []
        losss = []
        graph.train()
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY
            graph.flush()
            pred, loss = graph.forward(tX)[-2:]
            #print(loss)
            hatys.append(np.argmax(pred, axis=1))
            ys.append(tY)
            graph.backward()
            graph.optimstep(lr, wd1, wd2)
            losss.append(loss)
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)



