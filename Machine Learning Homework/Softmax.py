from numpy import *
from numpy import max as Max
from numpy import min as Min
import matplotlib.pyplot as plt
from random import sample as Sample
from sklearn.model_selection import KFold

# 类别数
c = 6

# 是否采用随机梯度下降法
is_random = 0

# 随机梯度下降中一次取得的样本数
batch_size = 100

# 迭代次数
iterate_num = 500

# 收敛阈值
lmt = 1e-6

# 点的大小
dot_size = 5

# 读取文件
file_path = "GMM" + str(c) + ".txt"
f = open(file_path)


# 读取数据
def load_data(f):
    lb = []
    x1 = []
    x2 = []
    n = 0
    for line in f.readlines():
        if n == 0:
            n += 1
            continue
        else:
            line = line.strip('\n').split('\t')
            lb.append(int(line[0]))
            x1.append(float(line[1]))
            x2.append(float(line[2]))
    return {'lb': lb, 'x1': x1, 'x2': x2}, len(lb)


# 最大-最小归一化处理
def max_min_normalization(x):
    # 一定要用numpy里的max和min，不然很joker
    x = (x - Min(x)) / (Max(x) - Min(x))
    return array(x)


# n为样本数量
gmm, nn = load_data(f)

# 对数据进行最大-最小归一化处理
xx1 = max_min_normalization(gmm['x1'])
xx2 = max_min_normalization(gmm['x2'])

yy = array([gmm['lb'][k] for k in range(nn)])

# 5折交叉检验
avg_acc = 0
kf = KFold(n_splits=5)
for train_index, test_index in kf.split(yy):
    # 学习率
    alpha = 1e-2
    # 准确率计算
    acc = 0
    # x是存放n个样本特征向量[1.0,x1,x2]的列表，x[k]表示第k个样本的特征向量
    # y是存放n个样本对应的真实标签，y[k]表示第k个样本的真实标签
    x1, x2, y = xx1[train_index], xx2[train_index], yy[train_index]
    test_x1, test_x2, test_y = xx1[test_index], xx2[test_index], yy[test_index]
    n = len(x1)
    test_n = len(test_x1)
    x = [mat([1.0, x1[k], x2[k]]).transpose() for k in range(n)]
    test_x = [mat([1.0, test_x1[k], test_x2[k]]).transpose() for k in range(test_n)]

    # 初始所有类别权重都为0.0
    # w[j]表示第j类对应的参数矩阵
    w = [zeros((3, 1)) for j in range(c)]


    # softmax函数
    def softmax(j, x):
        exp_sum = 0
        for j1 in range(c):
            exp_sum += exp(matmul(w[j1].transpose(), x))
        return exp(matmul(w[j].transpose(), x)) / exp_sum


    # 样本预测值
    def h(x):
        exp_sum = 0
        softmax_list = []
        for j1 in range(c):
            exp_sum += exp(matmul(w[j1].transpose(), x))
        for j in range(c):
            softmax_list.append(exp(matmul(w[j].transpose(), x)) / exp_sum)
        return argmax(softmax_list)


    # 损失函数
    def l():
        ll = 0
        for k in range(n):
            for j in range(c):
                ll += int(y[k] == j) * log(softmax(j, x[k]))
            return ll


    figure = plt.figure(figsize=(9.5, 2.8))
    ax1 = figure.add_subplot(1, 3, 1)
    ax2 = figure.add_subplot(1, 3, 2)
    ax3 = figure.add_subplot(1, 3, 3)
    ax1.set_title("Train Data")
    ax3.set_title("Loss")
    ax1.scatter(x1, x2, c=y, s=dot_size)
    ax3.set_xlim(0, iterate_num)
    # ax3.set_ylim(-2, -1)

    loss_list = []
    pre_labels = []
    # plt.ion()
    for tot in range(iterate_num):
        pre_labels.clear()
        loss = l()[0, 0]
        # print(loss)
        loss_list.append(loss)
        pre_labels = [h(x[k]) for k in range(n)]
        if is_random == 1:
            # 随机梯度下降
            ax2.set_title("Softmax_SGD")
            rand_index = Sample(range(0, n), batch_size)
            for j in range(c):
                s = zeros((3, 1))
                for k in range(batch_size):
                    index = rand_index[k]
                    s = s + (int(y[index] == j) - softmax(j, x[index])[0, 0]) * x[index]
                w[j] = w[j] + alpha * s

        else:
            # 梯度下降
            ax2.set_title("Softmax_GD")
            for j in range(c):
                s = zeros((3, 1))
                for k in range(n):
                    s = s + (int(y[k] == j) - softmax(j, x[k])[0, 0]) * x[k]
                w[j] = w[j] + alpha * s

        ax2.scatter(x1, x2, c=pre_labels, s=dot_size)
        plt.pause(0.01)
        ax2.cla()
        if tot > 0:
            ax3.plot([tot, tot + 1], [loss_list[tot - 1], loss_list[tot]], c='black')
            if abs(loss_list[tot] - loss_list[tot - 1]) < lmt:
                break
    ax2.scatter(x1, x2, c=pre_labels, s=dot_size)
    plt.close()
    # plt.ioff()
    # plt.show()
    # for ww in w:
    #     print(ww)
    for k in range(test_n):
        if test_y[k] == h(test_x[k]):
            acc += 1
    acc /= test_n
    print("acc=", acc)
    avg_acc += acc
avg_acc /= 5
print("avg_acc=", avg_acc)
