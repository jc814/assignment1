from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """
    一个两层(单隐层)的全联接网络类。该网络的输入层单元数N，隐层单元数为H，输出层单元数/分类类别数为C。

    我们训练神经网络使用softmax损失(严格说是交叉熵损失,可以把softmax函数理解为输出层的激活函数，不过实现时通过把softmax函数和计算交叉熵损失封装到一起)和在权重矩阵上的L2正则化。隐层使用ReLU非线性激活函数。


    该神经网络有如下的结构:

    input - fully connected layer - ReLU - fully connected layer - softmax

    第2个全联接层的输出是各个类别的得分.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        实例化类对象/网络模型的时候会调用init方法，并传入参数。
        初始化模型。权重被初始化为小的随机数偏置被初始化为0.权重和偏置存储在变量self.params中，采用字典，键的形式如下：

        W1: 第一层的权重 维度(D, H)
        b1: 第一层的偏置 维度 (H,)
        W2: 第二层的权重 维度 (H, C)
        b2: 第二层的偏置 维度(C,)

        Inputs:
        - input_size: 输入层单元数/样本的特征向量维数.
        - hidden_size: 隐层单元数.
        - output_size: 输出层单元数/分类类别数.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        计算两层全联接网络的loss和梯度。

        注意计算损失的公式和之前吴恩达专项课程中稍有差别，因为之前我们把样本真实标签转换成了one-hot形式，如进行3分类，标签为2(类别索引)，则表示为[0,0,1]；本实验中我们并没有把标签转换为one-hot形式，直接使用标签/类别索引[0,C)之间的一个整数，所以损失函数的定义稍有差别，损失函数部分的前向/反向传播稍有不同，原理是一致的。

        Inputs:
        - X: 数据集样本的特征矩阵 (N, D). 每一行代表一个样本的特征向量.
        - y: 数据集样本的标签. y[i] 是样本 X[i]的标签,  y[i] 是一个整数 0 <= y[i] < C.         这个参数是缺省的,如果没有传入默认为None，此时只返回类别得分, 如果传入了则返回loss和梯度。

        - reg: 正则化强度，默认为0不进行正则化.

        Returns:
        如果没传入y，y=None, 返回一个得分矩阵 维度(N, C) 每一行代表一个样本的得分向量。通常对mini-batch中的N个样本同时进行计算(矢量化并行计算).
        如果传入y，则返回一个元组。包括以下几部分:
        - loss: mini-batch上N个样本的损失(数据损失和正则化损失)
        - grads:字典形式，键和self.params一致，为权重和偏置参数的名称(字符串),值为该参数相对于损失函数的梯度。

        """
        # 从参数字典中取出初始化的参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # 前向传播：计算得分
        scores = None

        z1 = X.dot(W1) + b1 #(N,H)
        h1 = np.maximum(0,z1) #(N,H)
        z2 = h1.dot(W2) + b2  #(N,C)
        scores = z2


        if y is None:
            return scores

        # 计算loss
        loss = None

        maxLogC = np.max(scores,axis=1).reshape((N,1)) #scores中每一行代表一个样本的得分向量 找到每一行中的最大值  rshape:(N,) -> (N,1) 满足广播规则
        scores = scores - maxLogC #广播 (N,C)-(N,1) 每一行减去其所在行最大值 使其最大值为0，避免指数爆炸
        expScores = np.exp(scores)
        loss = np.sum(-np.log(expScores[np.arange(N),y]/np.sum(expScores,axis=1)))
        loss /= N  #计算N个样本的平均数据损失
        loss += reg*(np.sum(W1*W1)+np.sum(W2*W2)) #加上正则化损失 一般只对权重进行惩罚，如果对偏置也进行惩罚对结果几乎没有影响，但习惯不那么做
        #reg前可以乘以0.5 也可以不乘


        # 反向传播：计算梯度
        grads = {}

        dh2 = expScores/np.sum(expScores,axis=1,keepdims=True) #keepdims保持维度 用2维数组表示结果 (N,C)
        dh2[np.arange(N),y] -= 1 #(N,C)
        dh2 /= N #计算N个样本的平均梯度 (N,C)

        #前向传播分阶段计算 反向传播时可以使用其中间结果
        dW2 =  h1.T.dot(dh2)   #(H,C) 数据梯度
        dW2 += 2*reg*W2     #正则化梯度  如果之前正则化损失*0.5 就会约掉这个2  我们之前没有乘0.5  所以会有一个平方项求导产生的2
        db2 = np.sum(dh2,axis=0) #(C,)
        dh1 = dh2.dot(W2.T)          #(N,H)

        #ReLu Max函数
        dz1 = dh1 #(N,H)
        dz1[h1<=0] = 0  #(N,H)

        dW1 = X.T.dot(dz1)   #(D,H)数据梯度
        dW1 += 2*reg*W1 #正则化梯度  如果之前正则化损失*0.5 就会约掉这个2  我们之前没有乘0.5  所以会有一个平方项求导产生的2
        db1 = np.sum(dz1,axis=0) #(H,)

        grads['W2']=dW2
        grads['b2']=db2
        grads['W1']=dW1
        grads['b1'] = db1

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        """
        使用mini-batch梯度下降训练神经网络.

        Inputs:
        - X:训练样本的特征矩阵(N,D) 每一行代表一个样本的特征向量
        - y:训练样本的标签 维度(N,) y[i] = c 意味着样本
          X[i] 的标签是c, 其中c是一个整数 0 <= c < C.
        - X_val: 验证样本的特征矩阵(N_val,D) 每一行代表一个样本的特征向量
        - y_val: 验证样本的标签 维度(N,) y_val[i] = c 意味着样本
          X_val[i] 的标签是c, 其中c是一个整数 0 <= c < C.
        - learning_rate: 实数，学习率
        - learning_rate_decay: 实数 学习率衰减率 随着梯度下降迭代的进行学习率应该逐渐变小，每一个epoch(完整遍历一遍训练集)进行一次衰减
        - reg: 实数 正则化强度.
        - num_iters: 梯度下降迭代次数.
        - batch_size: mini-batch中包含的样本数  每次梯度下降迭代使用的样本数.
        - verbose: 为true 打印优化进程.
        """
        num_train = X.shape[0]  #训练样本数
        iterations_per_epoch = max(num_train / batch_size, 1) #一个epoch包含的mini-batch数

        # 使用mini-batch GD优化模型参数
        loss_history = []  #存放训练过程中的损失
        train_acc_history = [] #存放训练过程中模型在训练集上的准确率
        val_acc_history = [] #存放模型在验证集上的准确率

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            #每次从训练集X中随机有放回的取样batch_size个样本 作为一个mini-batch
            randomIndex = np.random.choice(len(X),batch_size,replace=True)
            X_batch = X[randomIndex]
            y_batch = y[randomIndex]

            # 使用当前的mini-batch计算loss和梯度
            loss,grads = self.loss(X_batch,y_batch,reg=reg)
            loss_history.append(loss)  #保留每个mini-batch上的loss


            #使用梯度下降法更新参数
            for param_name in self.params: #遍历每一个参数
                self.params[param_name] -= learning_rate*grads[param_name]


            if verbose and it % 1000 == 0: #每1000次迭代(1000个mini-batch)打印一次loss
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # 每训练一个epoch检查一次模型在训练集和验证集上的准确率 进行一次学习率衰减.
            #一个epoch包含 num_train // batch_size个mini-batch
            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # 对学习率进行衰减
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        使用两层神经网络训练的权重/参数来预测数据集样本的标签，对于每一个样本(数据点，可以看作高维空间中的一个点)我们预测出在各个类别上的得分，最高得分对应的类别索引就是该样本的标签。


        Inputs:
        - X: 数据集 (N, D) 每一行代表一个样本的特征向量 包含N个样本/数据点

        Returns:
        - y_pred: 一维数组 维度 (N,) 包含对X中的每个样本预测的标签. y_pred[i] = c意味着为样本 X[i] 预测的标签/类别索引是 c, c是一个整数 0 <= c < C.
        """
        y_pred = None

        #取出训练的模型参数
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        z1 = X.dot(W1) + b1 #(N,H)
        h1 = np.maximum(0,z1) #(N,H)

        z2 = h1.dot(W2) + b2 #(N,C) 每一行代表一个样本的得分向量

        y_pred = np.argmax(z2,axis=1) #计算每行最大值的索引 (N,)

        return y_pred


