# 四川大学模式识别与机器学习课程平时作业
## 概述
本项目是针对模式识别课程的平时作业，主要包括监督学习和无监督学习两部分。我们将实现基本的分类和聚类算法，并对结果进行可视化展示。
## 内容
### 1. 平时作业（上机）
#### （1）监督学习
- 随机产生两类样本，并将其分为测试集和训练集。
- 分别使用感知器、支持向量机 (SVM) 和前馈神经网络求判别函数。
- 编写程序运行，输出判别函数，并绘制分类结果图表。
#### （2）无监督学习
- 利用已知的100个样本（用两个高斯分布随机生成），每个样本包含两个特征（取值范围在0-10）。
- 使用C均值算法和DBSCAN算法进行分类，编程实现并绘制聚类结果图。

### 安装依赖
- Python 依赖库：`numpy`、`matplotlib`、`scikit-learn`
可通过清华镜像源安装：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
```