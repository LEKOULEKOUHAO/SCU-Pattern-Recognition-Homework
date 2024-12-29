import numpy as np
import matplotlib.pyplot as plt

# 生成两类样本
np.random.seed(0)  # 保证结果可重复
class1 = np.random.randn(100, 2) + np.array([2, 2])
class2 = np.random.randn(100, 2) + np.array([-2, -2])

# 合并样本并打乱顺序
data = np.concatenate((class1, class2))
labels = np.concatenate((np.ones(100), np.zeros(100)))
indices = np.random.permutation(200)
data = data[indices]
labels = labels[indices]

# 划分训练集和测试集 (80%训练，20%测试)
train_data = data[:160]
train_labels = labels[:160]
test_data = data[160:]
test_labels = labels[160:]

# 感知器实现
def Perceptron(X, y, epochs=100, learning_rate=0.1):
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    for _ in range(epochs):
        for i in range(n):
            z = np.dot(X[i], w) + b
            y_pred = 1 if z > 0 else 0
            if y_pred != y[i]:
                w += learning_rate * (y[i] - y_pred) * X[i]
                b += learning_rate * (y[i] - y_pred)
    return w, b

# 训练感知器
w_Perceptron, b_Perceptron = Perceptron(train_data, train_labels)

# 感知器判别函数
def Perceptron_predict(x, w, b):
    return 1 if np.dot(x, w) + b > 0 else 0


# 支持向量机实现
def SVM(X, y, epochs=100, learning_rate=0.1): 
    n, m = X.shape
    w = np.zeros(m)
    b = 0
    for _ in range(epochs):
        for i in range(n):
            z = np.dot(X[i], w) + b
            if (y[i] == 1 and z <= 0) or (y[i] == 0 and z >=0): #简化版SVM，仅考虑错误分类的情况
                w += learning_rate * y[i] * X[i]
                b += learning_rate * y[i]
    return w, b

# 训练SVM
w_svm, b_svm = SVM(train_data, train_labels)

# SVM判别函数
def svm_predict(x, w, b):
    return 1 if np.dot(x, w) + b > 0 else 0

# 前馈神经网络实现
def single_layer_nn(X, y, epochs=100, learning_rate=0.1): #单层神经网络
    n, m = X.shape
    w = np.random.rand(m)
    b = np.random.rand()
    for _ in range(epochs):
        for i in range(n):
            z = np.dot(X[i], w) + b
            y_pred = 1 / (1 + np.exp(-z))
            error = y[i] - y_pred
            w += learning_rate * error * y_pred * (1 - y_pred) * X[i]
            b += learning_rate * error * y_pred * (1 - y_pred)
    return w, b
# 训练前馈神经网络
w_nn, b_nn = single_layer_nn(train_data, train_labels)
# 前馈神经网络判别函数
def nn_predict(x, w, b):
    z = np.dot(x, w) + b
    return 1 if 1 / (1 + np.exp(-z)) > 0.5 else 0

# 边界
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 绘图函数
def plot_decision_boundary(w, b, title):
    Z = np.array([Perceptron_predict(np.array([x,y]), w, b) for x, y in zip(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')
    plt.scatter(class2[:, 0], class2[:, 1], label='Class 2')
    plt.title(title)
    plt.legend()
    plt.show()

# 绘制判别结果
plot_decision_boundary(w_Perceptron, b_Perceptron, "Perceptron")
plot_decision_boundary(w_svm, b_svm, "SVM")
plot_decision_boundary(w_nn, b_nn, "Single Layer NN")