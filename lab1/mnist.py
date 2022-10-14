import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F


n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
lr = 1e-3
lengths = (n_features, 1000, n_classes)


class Model(nn.Module):

    # TODO Design the classifier.

    def __init__(self,lengths) -> None:
        super().__init__()
        self.lengths = lengths
    def forward(self,x):
        self.L1 = nn.Linear(self.lengths[0],20)
        y1 = self.L1(x)
        self.s0 = F.Sigmoid()
        y1_ = self.s0(y1)
        self.L2 = nn.Linear(20,self.lengths[2])
        y3  = self.L2(y1_)
        self.S1 = F.Sigmoid()
        prob = self.S1(y3)
        
        return prob
        
    def backward(self, dy: np.ndarray) -> np.ndarray:
        self.dy0 = self.S1.backward(dy)
        self.dy1 = self.L2.backward(self.dy0)#grad for the last linear layer 
        self.dy2 = self.s0.backward(self.dy1)
        self.dy3 = self.L1.backward(self.dy1)#grad for first linear layer
        
    # End of todo


def load_mnist(mode='train', n_samples=None, flatten=True):
    images = 'train-images.idx3-ubyte' if mode == 'train' else 't10k-images.idx3-ubyte'
    labels = 'train-labels.idx1-ubyte' if mode == 'train' else 't10k-labels.idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape(
        (length, 28, 28)).astype(np.int32)
    if flatten:
        X = X.reshape(length, -1)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape(
        (length)).astype(np.int32)
    return (X[:n_samples] if n_samples is not None else X,
            y[:n_samples] if n_samples is not None else y)


def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all',
                       sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test'))
    model = Model(lengths)
    optimizer = nn.optim.SGD(model, lr=lr, momentum=0.9)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)
    
    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4 / bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            probs = model.forward(X/255)
            loss = criterion(probs, y)
            grad = criterion.backward()
            model.backward(grad)#这里要计算所有的梯度
            optimizer.step()#这里要更新梯度值
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f}'
                                f' loss={loss:.3f}')

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()
