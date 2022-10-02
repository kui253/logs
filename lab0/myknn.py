import numpy as np
class knnclassifier:
    def __init__(self):
        pass#do nothing
    #训练集就只需要赋值就可，后面来进行做差求出距离
    def train(self,X,y):
        self.Xtr = X
        self.ytr = y
    def predict(self,X): 
        num_test = X.shape[0]
        #这里的shape是指这个np数组的形状，比如4*9，shape[0]表示的这个数组描述形状的第一个数字
        
        Ypred = np.zeros(num_test,dtype = self.ytr.dtype)
        
        for i in range(num_test):
            #使用的是马氏距离，应该是k = 1 的情况
            distances = np.sum(np.abs(self.Xtr - X[i,:]),axis =1)
            #找出距离最小的的一个的下标
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]
        return Ypred#返回的是一个数组