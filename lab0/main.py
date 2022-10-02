import numpy as np
import matplotlib.pyplot as plt

from myknn import knnclassifier
import struct


def load_mnist(root='./mnist'):
 
    binfile1 = open('train-images.idx3-ubyte', 'rb') # 读取二进制文件
    binfile2 = open('train-labels.idx1-ubyte', 'rb')
    binfile3 = open('t10k-images.idx3-ubyte','rb')
    binfile4 = open('t10k-labels.idx1-ubyte','rb')
    buffers1 = binfile1.read()
    buffers3 = binfile3.read()
   
    head1 = struct.unpack_from('>IIII', buffers1, 0) # 取前4个整数，返回一个元组
    head3 = struct.unpack_from('>IIII', buffers3, 0)
    #说明这个数集的前四个并不是图像的信息，而是数据的整体信息
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置
    imgNum1 = head1[1]
    width = head1[2]
    height = head1[3]
    imgNum2 = head3[1]
    bits = imgNum1 * width * height  # data一共有60000*28*28个像素值
    bits2 = imgNum2* width * height 
    bitsString1 = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'一次读取这么多
    bitsString3 = '>' + str(bits2) + 'B'
    imgs_train = struct.unpack_from(bitsString1, buffers1, offset) # 取data数据，返回一个元组
    imgs_test = struct.unpack_from(bitsString3, buffers3, offset)
    binfile1.close()
    binfile3.close()
    imgs_train = np.reshape(imgs_train, [imgNum1, width * height]) # reshape为[60000,784]型数组
    imgs_test = np.reshape(imgs_test,[imgNum2, width * height])
    #此时这里的imgs还仅仅是是一个1*784的类型数组
    buffers2 = binfile2.read()
    buffers4 = binfile4.read()
    head2 = struct.unpack_from('>II', buffers2, 0) # 取label文件前2个整形数
    head4 = struct.unpack_from('>II', buffers4, 0)
    labelNum = head2[1]#这个是第二个数字，取出来是一个它的个数
    labelNum4 = head4[1]
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置

    numString2 = '>' + str(labelNum) + "B" # fmt格式：'>60000B'
    numString4 = '>' + str(labelNum4) + "B"
    labels_train = struct.unpack_from(numString2, buffers2, offset) # 取label数据
    labels_test = struct.unpack_from(numString4, buffers4, offset)
    binfile2.close()
    binfile4.close()
    labels_train = np.reshape(labels_train, [labelNum]) # 转型为列表(一维数组)
    labels_test = np.reshape(labels_test, [labelNum4])
    return imgs_train,labels_train,imgs_test,labels_test



def main():
    X_train, y_train, X_test, y_test = load_mnist()

    knn = knnclassifier()
    knn.train(X_train, y_train)
    y_pred = knn.predict(X_test)
    correct = sum((y_test - y_pred) == 0)#生成式？直接在求和里放入判断 = 求出满足这项的个数？

    print('==> correct:', correct)
    print('==> total:', len(X_test))
    print('==> acc:', correct / len(X_test))

    # plot pred samples
    '''
    some thing wrong with this 
    fig, ax = plt.subplots(nrows=4, ncols=5, sharex='all', sharey='all')
    fig.suptitle('Plot predicted samples')
    ax = ax.flatten()
    for i in range(20):
        img = X_test[i]
        ax[i].set_title(y_pred[i])
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    '''


if __name__ == '__main__':
    main()