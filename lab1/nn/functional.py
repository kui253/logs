import numpy as np
from .modules import Module
#todo

class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.
        self.out  = 1/(1+np.exp(-x))
        return self.out

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.
        dx = dy * self.out * (1-self.out)
        return dx
        # End of todo


class Tanh(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of tanh function.

        self.out = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        return self.out
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of tanh function.

        dx = dy * (self.out+1) * (1-self.out)
        return dx
        # End of todo


class ReLU(Module):#ReLU函数就是比较最后得到的值和0的关系

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.
        self.x0 = x
        self.out = np.maximum(0,x)
        return self.out
        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.

        dx = dy * np.where(self.x0>=0,1,0)
        return dx
        # End of todo


class Softmax(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of Softmax function.
        '''
        sum_SoftMax = sum(np.exp(x))
        self.out = np.exp(x)/sum_SoftMax
        return self.out
        '''
        
        ex = np.exp(x)
        sum_ex = np.sum(ex)
        self.solfmax = ex / sum_ex
        return self.solfmax

        # End of todo

    def backward(self, dy):

        # Omitted.
        pass


class Loss(object):
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     probs = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, probs, targets):
        self.probs = probs
        self.targets = targets
        print("error for parent __call__")
        return self

    def backward(self):
        print("error for parent backward")
        


class SoftmaxLoss(Loss):

    def __call__(self, probs, targets):#这里的probs是直接通过softmax正向传播得到概率，相当于是S_i

        # TODO Calculate softmax loss.
        self.y0 = targets# this is the labels
        self.s0 = probs# this is the compute probility
        return -np.sum(np.log(probs)*targets)

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of softmax loss function.

        ds = -(self.y0/self.s0)
        return ds 

        # End of todo


class CrossEntropyLoss(Loss):

    def __call__(self, probs, targets):

        # TODO Calculate cross-entropy loss.

        self.p0 = probs
        x,y = probs.shape
        self.q0 = np.zeros([x,y])
        for i,j in enumerate(targets):
            self.q0[i,j] = 1 
        self.loss =  -np.sum(self.q0 * np.log(self.p0)+(1-self.q0)*np.log(1-self.p0))
        return self.loss/1000/10
        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of cross-entropy loss function.
        self.dx = (self.q0-self.p0)/(self.p0 * (1 - self.p0))
        return -self.dx/1000/10363
        # End of todo
