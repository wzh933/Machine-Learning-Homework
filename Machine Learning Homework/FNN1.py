from numpy import *
from numpy import max as Max
from numpy import min as Min
import matplotlib.pyplot as plt
from random import sample as Sample
from sklearn.model_selection import KFold
import matplotlib as mpl

# 类别数
c = 6

# 隐层节点数
q = 5
print("隐层节点数为:", q)

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
    # x = [mat([1.0, x1[k], x2[k]]).transpose() for k in range(n)]
    # test_x = [mat([1.0, test_x1[k], test_x2[k]]).transpose() for k in range(test_n)]
    x = [[1.0 for k in range(n)], [x1[k] for k in range(n)], [x2[k] for k in range(n)]]
    test_x = [[1.0 for k in range(test_n)], [test_x1[k] for k in range(test_n)], [test_x2[k] for k in range(test_n)]]

    # 初始化参数，均匀分布在(0,1)上
    v = random.random((3, q))
    w = random.random((q, c))


    def sigmoid(z):
        return 1 / (1 + exp(-z))


    def predict(x1, x2):
        pre_labels = []
        for k in range(len(x1)):
            a = zeros(q)
            for l in range(q):
                a[l] = v[0, l] + v[1, l] * x1[k] + v[2, l] * x2[k]
            h = sigmoid(a)
            # print("a=", a)
            # print("h=", h)
            o = zeros(c)
            for j in range(c):
                for l in range(q):
                    o[j] += w[l, j] * h[l]
            exp_o = exp(o)
            exp_sum = sum(exp_o)
            pre_y = zeros(c)
            for j in range(c):
                pre_y[j] = exp_o[j] / exp_sum
            pre_labels.append(argmax(pre_y))
        return pre_labels


    def pre_predict(x1, x2):
        a = zeros(q)
        for l in range(q):
            a[l] = v[0, l] + v[1, l] * x1 + v[2, l] * x2
        h = sigmoid(a)
        # print("a=", a)
        # print("h=", h)
        o = zeros(c)
        for j in range(c):
            for l in range(q):
                o[j] += w[l, j] * h[l]
        exp_o = exp(o)
        exp_sum = sum(exp_o)
        pre_y = zeros(c)
        for j in range(c):
            pre_y[j] = exp_o[j] / exp_sum
        pre_label = (argmax(pre_y))
        return pre_label


    figure = plt.figure(figsize=(9.5, 2.8))
    ax1 = figure.add_subplot(1, 3, 1)
    ax2 = figure.add_subplot(1, 3, 2)
    ax3 = figure.add_subplot(1, 3, 3)
    ax1.set_title("Train Data")
    ax3.set_title("Loss")
    ax1.scatter(x1, x2, c=y, s=dot_size)
    ax3.set_xlim(0, iterate_num)
    ax3.set_ylim(0, 2)
    lce_list = []
    for tot in range(iterate_num):
        ax1.set_title("Train Data")
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax2.set_title("FNN")
        sum_lce = 0
        pre_labels = predict(x1, x2)
        ax2.scatter(x1, x2, c=pre_labels, s=dot_size)
        # 把预测过程重新写一遍而不是直接用函数，目的是把中间变量保存下来用于后面的计算
        for k in range(n):
            a = zeros(q)
            for l in range(q):
                a[l] = v[0, l] + v[1, l] * x1[k] + v[2, l] * x2[k]
            h = sigmoid(a)
            # print("a=", a)
            # print("h=", h)
            o = zeros(c)
            for j in range(c):
                for l in range(q):
                    o[j] += w[l, j] * h[l]
            exp_o = exp(o)
            exp_sum = sum(exp_o)
            pre_y = zeros(c)
            for j in range(c):
                pre_y[j] = exp_o[j] / exp_sum

            # 交叉熵损失函数
            lce = 0
            ohy = zeros(c)
            ohy[y[k]] = 1
            for j in range(c):
                lce -= ohy[j] * log(pre_y[j])
            sum_lce += lce
            # print(lce)
            # lce_list.append(lce)

            # 参数更新
            for l in range(q):
                for j in range(c):
                    w[l, j] -= alpha * (pre_y[j] - ohy[j]) * h[l]

            for i in range(3):
                for l in range(q):
                    sum_loss = 0
                    for j in range(c):
                        sum_loss += (pre_y[j] - ohy[j]) * w[l, j]
                    v[i, l] -= alpha * h[l] * (1 - h[l]) * x[i][k] * sum_loss

        lce_list.append(sum_lce / n)
        if tot > 0:
            ax3.plot([tot - 1, tot], [lce_list[tot - 1], lce_list[tot]], c="black")

        N, M = 300, 300
        x1_min, x2_min = min(x1) - 0.5, min(x2) - 0.5
        x1_max, x2_max = max(x1) + 0.5, max(x2) + 0.5
        t1 = linspace(x1_min, x1_max, N)
        t2 = linspace(x2_min, x2_max, M)
        xxx1, xxx2 = meshgrid(t1, t2)
        x_show = stack((xxx1.flat, xxx2.flat), axis=1)
        one = ones((len(x_show), 1))
        x_show = c_[one, x_show]
        y_predict = []
        for i in range(0, len(x_show)):
            y_predict.append(pre_predict(x_show[i][1], x_show[i][2]))
        y_predict = array(y_predict)
        cm_light = mpl.colors.ListedColormap(
            ['#A0FFA0', '#FFA0A0', '#A0A0FF', '#FF16D5', '#2CAAE6', '#392EE6', '#E9F07C', '#608F0E'])
        ax1.pcolormesh(xxx1, xxx2, y_predict.reshape(xxx1.shape), cmap=cm_light, alpha=0.5)
        ax1.scatter(x1, x2, c=y, s=dot_size)
        plt.pause(0.01)
        ax1.cla()
        ax2.cla()
    plt.close()

    # 在训练集上进行检验
    pre_labels = predict(test_x1, test_x2)
    for k in range(test_n):
        if pre_labels[k] == test_y[k]:
            acc += 1
    acc /= test_n
    print("acc=", acc)
    avg_acc += acc
avg_acc /= 5
print("avg_acc=", avg_acc)
