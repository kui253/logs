from tkinter.messagebox import NO
from .tensor import *
from .modules import *


class Optim(object):

    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self._step_module(self.module)#_step_module是这个函数传入的，所以下面的module就是self.module

    def _step_module(self, module):

        # TODO Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.
        self._update_weight(module.L2.w)
        self._update_weight(module.L1.w)

        # End of todo

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module, lr, momentum: float=0):
        super(SGD, self).__init__(module, lr)
        self.momentum = momentum

    def _update_weight(self, tensor):
        
        # TODO Update the weight of tensor
        # in SGD manner.
        tensor -= self.lr * tensor.grad
        '''
        self.v = np.zeros(np.shape(tensor))
        self.v = self.momentum * self.v + tensor.grad
        tensor -= self.lr * self.v
        '''
        # End of todo


class Adam(Optim):

    def __init__(self, module, lr,betas = (0.9,0.999),eps = 1e-08):
        super(Adam, self).__init__(module, lr)
        self.beta1 = betas[0]
        # TODO Initialize the attributes
        # of Adam optimizer.

        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = None
        self.v = None
        self.n = 0

        # End of todo

    def _update_weight(self, tensor):

        # TODO Update the weight of
        # tensor in Adam manner.

        if self.m is None:
            self.m = np.zeros_like(tensor)
        if self.v is None:
            self.v = np.zeros_like(tensor)

        self.n += 1

        self.m = self.beta1 * self.m + (1 - self.beta1) * tensor.grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(tensor.grad)

        alpha = self.lr * np.sqrt(1 - np.power(self.beta2, self.n))
        alpha = alpha / (1 - np.power(self.beta1, self.n))

        tensor -= alpha * self.m / (np.sqrt(self.v) + self.eps)


        # End of todo
