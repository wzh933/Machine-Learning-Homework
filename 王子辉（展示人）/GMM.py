from numpy import *
from numpy import max as Max
from numpy import min as Min
import matplotlib.pyplot as plt
from random import sample as Sample
from sklearn.model_selection import KFold

# 类别数
c = 8
print("基于高斯分布假设的贝叶斯模型做分类任务的类别数是:", c)

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
test = 1
if test == 1:
    print("test:")
else:
    print("train:")
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
    x = [mat([x1[k], x2[k]]).transpose() for k in range(n)]
    test_x = [mat([test_x1[k], test_x2[k]]).transpose() for k in range(test_n)]

    pij = []
    u = []
    ep = []
    for j in range(c):
        sum = 0
        sum_up1 = zeros((2, 1))
        sum_up2 = zeros((2, 2))
        for k in range(n):
            if y[k] == j:
                sum += 1
                sum_up1 = sum_up1 + x[k]
        uj = sum_up1 / sum
        for k in range(n):
            if y[k] == j:
                sum_up2 = sum_up2 + matmul(x[k] - uj, (x[k] - uj).transpose())
        epj = sum_up2 / sum
        pij.append(sum / n)
        u.append(uj)
        ep.append(epj)


    # print(pij)
    # print(u)
    # print(ep)
    def pre(x, j):
        mul1 = matmul((x - u[j]).transpose(), linalg.inv(ep[j]))
        mul2 = matmul(mul1, x - u[j])
        exp_mul = exp(-1 / 2 * mul2)
        det = 2 * pi * sqrt(linalg.det(ep[j]))
        return pij[j] * exp_mul / det


    def pre_label(x):
        pre_list = [pre(x, j) for j in range(c)]
        return argmax(pre_list)


    if test == 1:
        for k in range(test_n):
            if pre_label(test_x[k]) == test_y[k]:
                acc += 1
        acc /= test_n
        print("acc=", acc)
    else:
        for k in range(n):
            if pre_label(x[k]) == y[k]:
                acc += 1
        acc /= n
        print("acc=", acc)
    avg_acc += acc
avg_acc /= 5
print("avg_acc=", avg_acc)
