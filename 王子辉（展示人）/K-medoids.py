from numpy import *
from numpy import max as Max
from numpy import min as Min
import matplotlib.pyplot as plt

# 类别数
c = 6

# 迭代次数
iterate_num = 200

# 学习率
alpha = 1e-2

# 收敛值
lmt = 1e-3

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
    return x


# n为样本数量
gmm, n = load_data(f)

# 轮盘次数
rand_num = 2 * n

# 对数据进行最大-最小归一化处理
x1 = max_min_normalization(gmm['x1'])
x2 = max_min_normalization(gmm['x2'])

# 初始所有类别权重都为0.0
# w[j]表示第j类对应的参数矩阵
w = [zeros((3, 1)) for j in range(c)]

# x是存放n个样本特征元组的列表，x[k]表示第k个样本的特征元组
# y是存放n个样本对应的真实标签，y[k]表示第k个样本的真实标签
x = [{'x1': x1[k], 'x2': x2[k]} for k in range(n)]
y = [gmm['lb'][k] for k in range(n)]


# plt.scatter(x1, x2, c="gray")
# plt.show()

def distance(triple1, triple2):
    return (triple1[0] - triple2[0]) ** 2 + (triple1[1] - triple2[1]) ** 2


def cal_medoid(cluster):
    sad = []
    for i in range(len(cluster)):
        sum_dis = 0
        for j in range(len(cluster)):
            sum_dis += distance(cluster[i], cluster[j])
        sad.append(sum_dis)
    return cluster[argmin(sad)]


# 随机选取样本点
# m = [(random.choice(x)['x1'], random.choice(x)['x2']) for k in range(c)]

# 选取样本点的改进
# 先选取一个随机样本点作为初始中心点
m = [(random.choice(x)['x1'], random.choice(x)['x2'])]

# 随机选取样本点
rand_m = [(random.choice(x)['x1'], random.choice(x)['x2']) for k in range(c)]

for k in range(c - 1):
    d = []
    for i in range(n):
        di = []
        for j in range(len(m)):
            di.append(distance((x1[i], x2[i]), m[j]))
        d.append(min(di))
    # print(d)
    # 得到各个点的概率
    d = [dd / sum(d) for dd in d]
    # print(d)
    # ans = argmax(d)
    # 进行轮盘赌
    sel_num = [0 for i in range(n)]
    for lun in range(rand_num):
        sum_d = [d[0]]
        for i in range(1, len(d)):
            sum_d.append(d[i] + sum_d[i - 1])
        # 生成0,1的随机数
        rand = random.random()
        # 待选取的点的下标ans
        ans = len(d) - 1
        for i in range(0, len(d)):
            if rand < sum_d[i]:
                ans = i
                break
        sel_num[ans] += 1
    ans = argmax(sel_num)
    m.append((x[ans]['x1'], x[ans]['x2']))
# print(m)

# 样本点类别列表
cc = [{'x1': [], 'x2': []} for k in range(c)]

# 样本点类别列表，cluster[i]中装的是第i类的点元祖
clusters = [[] for k in range(c)]

# 绘制子图
figure = plt.figure(figsize=(9.5, 2.8))
ax0 = figure.add_subplot(1, 3, 1)
ax1 = figure.add_subplot(1, 3, 2)
ax2 = figure.add_subplot(1, 3, 3)
ax0.set_title("Train Data")
ax2.set_title("WCSS")
ax0.scatter(x1, x2, c=y, s=dot_size)

wcss_list = []
labels = []
plt.ion()
for tot in range(iterate_num):
    labels.clear()
    for j in range(c):
        cc[j]['x1'].clear()
        cc[j]['x2'].clear()
    ax1.set_title("K-medoids")
    ax2.set_xlim(0, iterate_num)
    wcss = 0
    for i in range(n):
        # 得到样本点与各个类中心点的距离，并存储在列表中
        dis = [distance((x1[i], x2[i]), m[k]) for k in range(c)]

        # 根据欧氏距离划分样本集，对其进行分类并定标签
        # 标签即为列表dis的最小元素的下标k=1,...,c
        label = argmin(dis)

        # 累加平方欧式距离
        wcss += min(dis)

        # 对样本点进行分配
        # 处理成这样是为了方便求x1和x2的均值
        cc[label]['x1'].append(x1[i])
        cc[label]['x2'].append(x2[i])

        clusters[label].append((x1[i], x2[i]))

        # 颜色列表
        labels.append(label)

    wcss_list.append(wcss)
    ax1.scatter(x1, x2, c=labels, s=dot_size)
    # 绘制类中心点
    for k in range(c):
        ax1.scatter(m[k][0], m[k][1], c="black", marker="x")
    plt.pause(1)
    ax1.cla()
    # print(wcss)
    # 更新
    # 体现了方便之处
    # m = [(mean(cc[k]['x1']), mean(cc[k]['x2'])) for k in range(c)]
    m = [cal_medoid(clusters[k]) for k in range(c)]
    if tot > 0:
        ax2.plot([tot, tot + 1], [wcss_list[tot - 1], wcss_list[tot]], c='black')
        if abs(wcss_list[tot] - wcss_list[tot - 1]) < lmt:
            break
# 最终的分类结构可视化
ax1.scatter(x1, x2, c=labels, s=dot_size)
ax1.set_title("K-medoids")
plt.ioff()
plt.show()
