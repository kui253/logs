# Logs

### lab0（已完成）

#### kNN的算法

k近邻（kNN）算法的工作机制比较简单，

1. 根据某种距离测度找出距离给定待测样本距离最小的k个训练样本，根据k个训练样本进行预测。
2. 分类问题：k个点中出现频率最高的类别作为待测样本的类别
3. 回归问题：通常以k个训练样本的平均值作为待测样本的预测值
4. kNN模型三要素：距离测度、k值的选择、分类或回归决策方式

在lab0中使用的是k = 1 的情况，预测的结果有0.9631,knn类如下所示

[KNN算法](https://github.com/kui253/logs/blob/master/lab0/myknn.py))

[通过马氏距离来分类MINST数据集](https://github.com/kui253/logs/blob/master/lab0/myknn.py))

#### MINST数据集的处理

使用自带的open就可打开二进制的文件，然后最重要的就是设置偏移量了 ，这就需要看清楚idx1-ubyte的说明了，使用unpack_from可以截取二进制文件的文件的位置，具体的说明在代码中有详细的注释

[使用struct库读取二进制文件和处理偏移量](https://github.com/kui253/logs/blob/master/lab0/myknn.py))



### lab1（完成了helloworld）

import .xxxx 这里的.是使用的是相对路径

#### 根据logistic回归的求解过程

1. 逻辑回归的梯度下降问题，相当于就是一个反复链式求导问题，这里的命名规则就是，对某个元素var求偏导记成dvar，

2. 尽量使用numpy中的向量乘积形式，这样的计算速度更快

   ```python
   import numpy as np
   np.dot(a,b)#将两个矩阵点乘
   np.zeros(m,n)#初始化一个矩阵为0，然后往这里面添加元素就可以了
   #下面dw是一个矩阵
   dw += elem#equ dw.append(elem)
   ```

3. 正向传播和反向传播

   反向就是一个链式求导的过程，正向就是推导的过程，直接计算
   
   线性分类器，使用了逻辑回归来将计算的结果转换成概率，用得到的概率和label值作差得到dy，利用dy将进行反向传播
   
   [helloworld中的部分](https://github.com/kui253/logs/blob/master/lab1/helloworld.py)
   
   [linear部分的代码](https://github.com/kui253/logs/blob/master/lab1/nn/modules.py)



