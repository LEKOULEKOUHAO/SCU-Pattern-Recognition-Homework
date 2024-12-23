import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# 随机生成两类样本
np.random.seed(0)  # For reproducibility
class1 = np.random.randn(50, 2) + np.array([2, 2])  # 类1
class2 = np.random.randn(50, 2) + np.array([7, 7])  # 类2
# 合并样本并生成标签
X = np.vstack((class1, class2))
y = np.hstack((np.zeros(50), np.ones(50)))
# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 绘图函数
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                         np.linspace(ylim[0], ylim[1], 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.2, cmap='spring')

# 感知器实现
class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_features = X.shape[1]
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                update = self.lr * (y[index] - y_predicted)
                self.weights += update * x_i
                self.bias += update
    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)
# 使用感知器进行训练和预测
model = Perceptron()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 可视化
plot_decision_boundary(X, y, model)
plt.title("Perceptron Decision Boundary")
plt.show()

#支持向量机SVM实现
class SVM:
    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            for index, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                if y[index] * linear_output < 1:
                    self.weights -= self.lr * (self.lambda_param * self.weights - y[index] * x_i)
                    self.bias -= self.lr * y[index]
                else:
                    self.weights -= self.lr * self.lambda_param * self.weights
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.sign(linear_output)
# 使用SVM进行训练和预测
model = SVM()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 可视化
plot_decision_boundary(X, y, model)
plt.title("SVM Decision Boundary")
plt.show()

#前馈神经网络实现
class NeuralNetwork:
    def __init__(self, layers=[2, 4, 1], learning_rate=0.01, n_iters=1000):
        self.layers = layers
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = [np.random.randn(self.layers[i], self.layers[i+1]) for i in range(len(self.layers)-1)]
        self.biases = [np.random.randn(self.layers[i+1]) for i in range(len(self.layers)-1)]
    def fit(self, X, y):
        for _ in range(self.n_iters):
            for x_i, y_i in zip(X, y):
                a = self._forward_pass(x_i)
                self._backward_pass(a, y_i)
    def _forward_pass(self, x):
        a = [x]
        for i in range(len(self.layers)-1):
            a.append(self._activation(np.dot(a[i], self.weights[i]) + self.biases[i]))
        return a
    def _backward_pass(self, a, y):
        error = y - a[-1]
        for i in range(len(self.layers)-2, -1, -1):
            self.weights[i] += self.lr * np.outer(a[i], error)
            self.biases[i] += self.lr * error
            error = np.dot(self.weights[i], error)
    def _activation(self, x):
        return 1 / (1 + np.exp(-x))
    def predict(self, X):
        a = self._forward_pass(X)
        return np.where(a[-1] >= 0.5, 1, 0)
# 使用神经网络进行训练和预测
model = NeuralNetwork()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# 可视化
plot_decision_boundary(X, y, model)
plt.title("Neural Network Decision Boundary")
plt.show()