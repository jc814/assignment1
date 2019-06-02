from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

#从cs231n/classifiers/neural_net.py中导入TwoLayerNet类
from cs231n.classifiers.neural_net import TwoLayerNet

#绘图默认设置
plt.rcParams['figure.figsize'] = (10.0, 8.0) # 默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# 更多重载外部Python模块的魔法命令查看 http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython


def rel_error(x, y):
    """ 返回相对误差 """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

#创建一个小的网络模型和一些小型数据来检查我们的实现的正确性
#最后把正确实现的模型在整个数据集上进行训练
#注意我们为可重复的实验设置随机种子(保证每次生成的随机数一致)

input_size = 4   #输入层单元数  样本的特征向量维数
hidden_size = 10 #隐层单元数
num_classes = 3  #输出层单元数 分类的类别数
num_inputs = 5   #输入的样本数

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()

scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# 2者的差别会非常小. 结果应该 < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))

loss,_ = net.loss(X,y,reg=0.05)
print(loss)
correct_loss = 1.30378789133

# 差距会非常小 应该< 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

from cs231n.gradient_check import eval_numerical_gradient

# 使用梯度检查来验证反向传播的实现
# 如果实现是正确的，对于每一个参数W1, W2, b1, 和 b2其解析梯度和数值梯度之间的差别小于1e-8  .

loss, grads = net.loss(X, y, reg=0.05)

#  差别应该小于1e-8
for param_name in grads: #对于梯度字典grads中的每一个梯度 遍历键名/参数名
    #定义匿名函数f  他的输入是不同的参数W，输出在当前W下的损失值  内部调用了之前计算loss的函数
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    #下面是定义在cs231n/gradient_check.py中的梯度检查函数
    #比较每个参数(矩阵/向量)的数值梯度和解析梯度
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# 绘制训练过程中loss的变化
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

from cs231n.data_utils import load_CIFAR10


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    从硬盘中加载cifar-10数据集并预处理，为2层前联接网络分类器作准备. 采取的步骤和SVM
    实验中相同，只不过这里把它压缩成了一个函数.
    """
    # 加载原始cifar-10数据集
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # #验证集中的样本是原始训练集中的num_validation(1000)个样本
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]

    # 训练集是原始训练集中的前num_training(49000)个样本
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    # 测试集是原始测试集中的前num_test(1000)个样本
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 预处理 将每张图像(32,32,3)拉伸为一维数组(3072,)
    # 各个数据集从四维数组(m,32,32,3) 转型为2维数组(m,3072)
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    # 预处理 0均值化 减去图像每个像素/特征上的平均值
    # 基于训练数据 计算图像每个像素/特征上的平均值
    mean_image = np.mean(X_train, axis=0)
    # 各个数据集减去基于训练集计算的各个像素/特征的平均值
    # (m,3072) - (3072,)  广播运算
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test


# 运行上述函数 得到各个数据集
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

input_size = 32 * 32 * 3 #输入层单元数 样本的特征向量维数
hidden_size = 50  #隐层单元数
num_classes = 10 #输出层单元数 分类类别数
#实例化一个两层神经网络的对象
net = TwoLayerNet(input_size, hidden_size, num_classes)

# 训练这个神经网络
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# 训练好的模型在验证集上的准确率
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)