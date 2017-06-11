
import numpy as nm
# from matplotlib import pyplot as plt


# 从testSet.txt中读取数据
def loadDataSet():
    dataMat = []
    labelMat = []

    fr = open('simple_testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))

    return dataMat, labelMat


def sigmoid(inX):
    return 1.0/(1 + nm.exp(-inX))


# 梯度上升算法
# 输入：训练集数据矩阵，分类列表
# 输出：回归系数
def gradAscent(dataMatIn, classLabels):
    # 转成矩阵 dataMatrix: N x 3
    dataMatrix = nm.mat(dataMatIn)
    # 转成矩阵 labelMat: N x 1
    labelMat = nm.mat(classLabels).transpose()

    # shape: return shape of array，返回矩阵的行和列
    m, n = nm.shape(dataMatrix)

    alpha = 0.001
    maxCycles = 500

    # ones: 返回一个array，用1填充，每一行都是一个array
    # ones((行数，列数))
    # ones((2, 1)) 返回 array([[ 1.],
    #                         [ 1.]])
    weights = nm.ones((n, 1))

    # maxCycles总迭代次数
    for k in range(maxCycles):
        # 矩阵相乘
        t = dataMatrix * weights
        # 对矩阵的每个值做sigmoid
        h = sigmoid(t)
        error = (labelMat - h)                  # 向量减法

        # transpose()：矩阵转置，行列调换
        # alpha：步长
        #
        weights = weights + alpha * dataMatrix.transpose() * error

    return weights


# 绘制最佳拟合线
# def plotBestFit(weights):
#     dataMat, labelMat = loadDataSet()
#     dataArr = nm.array(dataMat)
#     n = nm.shape(dataArr)[0]
#     xcord1 = []
#     ycord1 = []
#     xcord2 = []
#     ycord2 = []
#
#     for i in range(n):
#         if int(labelMat[i]) == 1:
#             xcord1.append(dataArr[i,1])
#             ycord1.append(dataArr[i,2])
#         else:
#             xcord2.append(dataArr[i,1])
#             ycord2.append(dataArr[i,2])
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
#     ax.scatter(xcord2, ycord2, s=30, c='green')
#     x = nm.arange(-3.0, 3.0, 0.1)
#     y = (-weights[0]-weights[1]*x)/weights[2]
#
#     ax.plot(x, y)
#     plt.xlabel('X1')
#     plt.ylabel('X2')
#
#     plt.show()


def stocGradAscent0(dataMatrix, classLabels):
    m, n = nm.shape(dataMatrix)
    alpha = 0.01
    weights = nm.ones(n)   #initialize to all ones
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = nm.shape(dataMatrix)
    weights = nm.ones(n)   #initialize to all ones
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.0001    #apha decreases with iteration, does not
            randIndex = int(nm.random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

