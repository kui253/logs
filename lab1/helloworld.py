
import numpy as np
import matplotlib.pyplot as plt
import nn


def gen_data(num):
    x1 = np.random.multivariate_normal([-0.5, -2], [[1, .75], [.75, 1]], num)
    x2 = np.random.multivariate_normal([0.5, 2], [[1, .75], [.75, 1]], num)
    X = np.vstack((x1, x2)).astype(np.float32)
    y = np.hstack((np.zeros(num), np.ones(num)))
    return X, y


def plot_data(X, y):
    plt.scatter(X[y==0, 0], X[y==0, 1], color='red')
    plt.scatter(X[y==1, 0], X[y==1, 1], color='blue')
    plt.show()


def plot_clf(clf, X, y):
    xx, yy = np.meshgrid(np.linspace(-4, 4, 500), np.linspace(-4, 4, 500))
    xy = np.c_[xx.ravel(), yy.ravel()]
    Z = np.where(clf(xy) >= 0, 1, 0)
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0], colors='black')
    plot_data(X, y)


def main():
    X, y = gen_data(100)
    plot_data(X, y)

    model = nn.Linear(2, 1)
    for i in range(20):#更新20次应该就好了
        probs = model(X)#调用了__call__函数
        preds = np.where(probs >= 0.5, 1, 0)#这是预测结果，不代表真正的dy
        model.backward(probs - y.reshape(-1, 1))#表示全部转成一列，然后行数自动计算
        model.w -= 0.1 * model.w.grad
        print(f'acc: {np.sum(preds == y.reshape(-1, 1)) / len(y):.2f}')
    plot_clf(model, X, y)


if __name__ == '__main__':
    main()
