from re import S
from xmlrpc.server import DocXMLRPCRequestHandler
import numpy as np
from itertools import product
from . import tensor


class Module(object):
    """Base class for all neural network modules.
    """
    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        #pending
        return x
        

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        return dy

    def train(self):
        """Sets the mode of the module to training.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = True
        for attr in vars(self).values():#返回对象object的属性和属性值的字典对象，如果没有参数，就打印当前调用位置的属性和属性值
            if isinstance(attr, Module):
                Module.train()

    def eval(self):
        """Sets the mode of the module to eval.
        Should not be overridden by subclasses.
        """
        if 'training' in vars(self):
            self.training = False
        for attr in vars(self).values():
            if isinstance(attr, Module):
                Module.eval()


class Linear(Module):

    def __init__(self, in_length: int, out_length: int):
        """Module which applies linear transformation to input.

        Args:
            in_length: L_in from expected input shape (N, L_in).
            out_length: L_out from output shape (N, L_out).
        """
        self.Lin = in_length
        self.Lout = out_length
        # w[0] for bias(偏差值) and w[1:] for weight（权重值）
        self.w = tensor.tensor((in_length + 1, out_length))#这个w是随机生成的一个高斯正态分布的矩阵

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        # TODO Implement forward propogation
        # of linear module.
        self.x0 = x
        self.out = np.dot(self.x0,self.w[1:])+self.w[0]
       
        return self.out
        # End of todo

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        # TODO Implement backward propogation
        # of linear module.
        dx = np.zeros((dy.shape[0],self.Lin))
        self.w.grad = np.zeros((np.shape(self.w)[0],np.shape(self.w)[1]))
        dx = np.dot(self.w[1:],dy.T).T
        temp2 = np.dot(self.x0.T,dy)
        self.w.grad[0] = np.sum(dy,axis = 0)/dy.shape[0]/dy.shape[1]
        self.w.grad[1:] = temp2/dy.shape[0]/dy.shape[1]
        
        return dx
        # End of todo


class BatchNorm1d(Module):

    def __init__(self, length: int, momentum: float=0.9):
        """Module which applies batch normalization to input.

        Args:
            length: L from expected input shape (N, L).
            momentum: default 0.9.
        """
        super(BatchNorm1d, self).__init__()

        # TODO Initialize the attributes
        # of 1d batchnorm module.

        self.length = length
        self.momentum = momentum

        # End of todo

    def forward(self, x):
        """Forward propagation of batch norm module.

        Args:
            x: input of shape (N, L).
        Returns:
            out: output of shape (N, L).
        """

        # TODO Implement forward propogation
        # of 1d batchnorm module.
        eps = 0.0001
        self.x_mean = x.mean(axis = 0)
        self.x_var = x.var(axis = 0)
        self.x_norm = (x-self.x_mean)/np.sqrt(self.x_var+eps)
        pass
        # lack of gama and beta
        # End of todo

    def backward(self, dy):
        """Backward propagation of batch norm module.

        Args:
            dy: output delta of shape (N, L).
        Returns:
            dx: input delta of shape (N, L).
        """

        # TODO Implement backward propogation
        # of 1d batchnorm module.

        pass

        # End of todo


class Conv2d(Module):

    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):
        """Module which applies 2D convolution to input.

        Args:
            in_channels: C_in from expected input shape (B, C_in, H_in, W_in).B是这个图片有多少个，c表示深度，h表示高度，w表示宽度
            channels: C_out from output shape (B, C_out, H_out, W_out).
            kernel_size: default 3.
            stride: default 1.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of 2d convolution module.

        self.C_in = in_channels
        self.C_out = channels
        self.kernel = np.random.randn(self.C_out,self.C_in,kernel_size,kernel_size)
        #因此，如果输入图片的形状是(H, W, C)，想要生成(H, W, C')的输出图片，则应该有C'个形状为(f, f, C)的卷积核，或者说卷积核组的形状是(C', f, f, C)。
        self.kernel_size = kernel_size
       
        if isinstance(stride,tuple):
            self.stride = stride[0]
        else:
            self.stride = stride
        self.padding = padding
        if bias:
            self.bia = bias
            self.bias = np.random.randn()
        # End of todo

    def cal_new_ksize(self,sl, stride, ksize, paddingnum):#当padding不为0的时候更新需要滑动的步数,f 是kernel的尺寸
        #这里的sl是矩阵的某一维度上的长度
        return (sl + 2 * paddingnum - ksize) // stride + 1

    def forward(self, x0):
        """Forward propagation of convolution module.

        Args:
            x: input of shape (B, C_in, H_in, W_in).
        Returns:
            out: output of shape (B, C_out, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of 2d convolution module.
        self.x0 = x0
        bat,c_in,h_in,w_in = np.shape(x0)
        km,kn = np.shape(self.kernel)
        x = np.pad(x0, [(0,0),(0,0), (self.padding, self.padding), 
                               (self.padding, self.padding)])
        h_0 = self.cal_new_out(h_in,self.stride,km,self.padding)
        w_0 = self.cal_new_out(w_in,self.stride,km,self.padding)
        #这里还是self.c_out的原因是这个kernel只有一个
        self.out = np.zeros([bat,self.C_out,h_0,w_0])
        for b in range(bat):#使用cal_new_out来计算的要进行平移几次
            for i_h in range(h_0):
                for i_w in range(w_0):
                    for c_i in range(self.C_out):
                        
                        h_lower = i_h * self.stride
                        h_upper = i_h * self.stride + self.ksize
                        w_lower = i_w * self.stride
                        w_upper = i_w * self.stride + self.ksize

                        #这里是切片的矩阵，然后将它和卷积核相乘
                        input_slice = x[b,:,h_lower:h_upper,w_lower:w_upper]
                        input_kslice = self.kernel[c_i]#循环卷积核就可以了
                        self.out[b,c_i,i_h,i_w] = np.sum(input_slice * input_kslice)
                        if self.bia:
                            self.out[b,c_i,i_h,i_w] += self.bias[c_i]
                        
        return self.out
        

        # End of todo

    def backward(self, dy):
        """Backward propagation of convolution module.

        Args:
            dy: output delta of shape (B, C_out, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C_in, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of 2d convolution module.
        #由于这里的卷积核的层数为1 ，所以这些c_in  = c_out
        dx = np.zeros(np.shape(self.x0))
        b,c_i,h_i,w_i, = np.shape(dy)#这里的切块的大小应该和dy保持一致
        for b_o in range(b):
            for h_o in range(h_i):
                for w_o in range(w_i):
                    for c_o in range (c_i):
                        h_lower = h_o * self.stride
                        h_upper = h_o * self.stride + self.ksize
                        w_lower = w_o * self.stride
                        w_upper = w_o * self.stride + self.ksize
                        dx[b_o,c_o,h_lower:h_upper,w_lower:w_upper] += self.kernel*dy[b,c_o,h_o,w_o]
        #如果原先有填充，那就将中间的部分挖出
        if self.padding>0:
            dx1 = dx[:,:,self.padding:-self.padding,self.padding:-self.padding]
        else: 
            dx1 = dx
        return dx1
        # End
        #  of todo


class Conv2d_im2col(Conv2d):#将卷积核中的每一列都展开，同时，将特征矩阵中与卷积和对应的部分也放成一列
    
    def __init__(self, in_channels: int, channels: int, kernel_size: int=3,
                 stride: int=1, padding: int=0, bias: bool=True):

        super().__init__(in_channels, channels,
         kernel_size,stride, padding, bias)#在调用父类的函数的时候不用再init里面添加self  

    def forward(self, x):

        # TODO Implement forward propogation of
        # 2d convolution module using im2col method.
        if self.padding:
            x = np.pad(x, [(0,0),(0,0), (self.padding, self.padding), 
                               (self.padding, self.padding)])
        km ,kn = np.shape(self.kernel)
        B_temp,c_out,xm,xn = np.shape(x)
        row_num = xm - km +1
        col_num = xn - kn + 1
        outmat = np.zeros([row_num*col_num,km*kn])#最后需要转置的
        out = np.zeros([self.x0.shape[0],row_num*col_num])
        for b in range(B_temp):
            for i_h in range(row_num):
                for i_w in range(col_num):
                    for c_i in range(c_out):
                        h_lower = i_h * self.stride
                        h_upper = i_h * self.stride + self.ksize
                        w_lower = i_w * self.stride
                        w_upper = i_w * self.stride + self.ksize
                        c_lower = c_i
                        c_upper = c_i + 1
                        input_slice = x[b,c_lower:c_upper,h_lower:h_upper,w_lower:w_upper]
                        input_slice = np.reshape(-1,km*kn)
                        outmat[i_h+i_w,:] = input_slice
            outmat = outmat.T
            kern = self.kernel.reshape(-1,km*kn)
            out[b,:] = np.dot(kern,outmat)
        return  out
        # End of todo


class AvgPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies average pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of average pooling module.
        self.kernel_size = kernel_size
        self.stride  = stride
        self.padding = padding
        

        # End of todo

    def forward(self, x):
        """Forward propagation of average pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of average pooling module
        self.x0 = x
        if self.padding:
             self.x = np.pad(x, [(0,0),(0,0), (self.padding, self.padding), 
                               (self.padding, self.padding)])
        else:
            self.x = x

        self.B = self.x.shape[0]
        self.C = self.x.shape[1]
        self.in_height = self.x.shape[2]
        self.in_width = self.x.shape[3]

        self.out_height = (self.in_height - self.kernel_size) // self.stride + 1
        self.out_width = (self.in_width - self.kernel_size) // self.stride + 1
        out = np.zeros([self.B,self.C,self.out_height, self.out_width])
        for b in range(self.B):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    for c in range(self.C):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel_size
                        end_j = start_j + self.kernel_size
                        out[b,c,i,j] = np.mean(x[b,c,start_i: end_i, start_j: end_j])
        return out

        # End of todo

    def backward(self, dy):
        """Backward propagation of average pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            dx: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of average pooling module.

        #这是没有padding的版本
        dx = np.zeros_like(self.x0)
        b_temp,c_temp, h_i,w_i = dy.shape
        for b in range(b_temp):
            for i in range(h_i):
                for j in range(w_i):
                    for c in range(c_temp):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel_size
                        end_j = start_j + self.kernel_size
                        dx[b,c,start_i: end_i, start_j: end_j] += dy[b,c,i, j] / (self.kernel_size * self.kernel_size)
        return dx
        #这里需要采用+=方法
        # End of todo


class MaxPool(Module):

    def __init__(self, kernel_size: int=2,
                 stride: int=2, padding: int=0):
        """Module which applies max pooling to input.

        Args:
            kernel_size: default 2.
            stride: default 2.
            padding: default 0.
        """

        # TODO Initialize the attributes
        # of maximum pooling module.
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        

        # End of todo

    def forward(self, x):
        """Forward propagation of max pooling module.

        Args:
            x: input of shape (B, C, H_in, W_in).
        Returns:
            out: output of shape (B, C, H_out, W_out).
        """

        # TODO Implement forward propogation
        # of maximum pooling module.
        self.x0 = x
        if self.padding:
            self.x = np.pad(x, [(0,0),(0,0),(self.padding, self.padding), 
                               (self.padding, self.padding)])
        else:
            self.x = x
       
        self.B = self.x.shape[0]
        self.C = self.x.shape[1]
        self.in_height = self.x.shape[2]
        self.in_width = self.x.shape[3]
        self.arg_max = np.zeros([self.B,self.C,self.in_height,self.in_width])
        self.out_height = (self.in_height - self.kernel_size) // self.stride + 1
        self.out_width = (self.in_width - self.kernel_size) // self.stride + 1
        out = np.zeros([self.B,self.C,self.out_height, self.out_width])
        for b in range(self.B):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    for c in range(self.C):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel_size
                        end_j = start_j + self.kernel_size
                        out[b,c,i,j] = np.max(x[b,c,start_i: end_i, start_j: end_j])
                        self.arg_max[b,c,i,j] = np.argmax(x[b,c,start_i: end_i, start_j: end_j])
        return out


        # End of todo

    def backward(self, dy):
        """Backward propagation of max pooling module.

        Args:
            dy: output delta of shape (B, C, H_out, W_out).
        Returns:
            out: input delta of shape (B, C, H_in, W_in).
        """

        # TODO Implement backward propogation
        # of maximum pooling module.

        dx = np.zeros(self.x.shape)
        
        for b in range(self.B):
            for i in range(self.out_height):
                for j in range(self.out_width):
                    for c in range(self.C):
                        start_i = i * self.stride
                        start_j = j * self.stride
                        end_i = start_i + self.kernel_size
                        end_j = start_j + self.kernel_size
                        index = np.unravel_index(self.arg_max[b,c,i, j], [self.kernel_size,self.kernel_size])
                        dx[b,c,start_i: end_i, start_j: end_j][index] = dy[b,c,i, j]#找出每个小方块的最大值，其他的都是0
        return dx


        # End of todo


class Dropout(Module):

    def __init__(self, p: float=0.5):

        # TODO Initialize the attributes
        # of dropout module.

        self.dropout_ratio = p


        # End of todo

    def forward(self, x):

        # TODO Implement forward propogation
        # of dropout module.

        self.mask = np.random.rand(*x.shape)>self.dropout_ratio
        out = x * self.mask / (1.0 - self.dropout_ratio)
        return out
        # End of todo

    def backard(self, dy):

        # TODO Implement backward propogation
        # of dropout module.

        dx = dy * self.mask / (1.0 - self.dropout_ratio)
        return dx
        # End of todo


if __name__ == '__main__':
    import pdb; pdb.set_trace()
