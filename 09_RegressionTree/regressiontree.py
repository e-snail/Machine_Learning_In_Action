
import numpy as np


def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')      # 使用\t分隔符切分每行的数据
        fltLine = map(float, curLine)           # map(function, sequence) ：对sequence中的item依次执行function(item)，将执行结果组成一个List
        # print(list(fltLine))
        dataMat.append(list(fltLine))

    return dataMat


# 将矩阵dataSet做二元切分，按照指定特征(第feature个)是否大于或小于value来划分
# 输入
#   dataSet
#   feature     待切分的特征（矩阵的一行是一个向量，向量的每个值是一个特征）
#   value       该特征的某个值
# 输出
#   dataSet中的第feature列中大于value的值所在的行数  从dataSet中取出对应的行（第一个）
#   dataSet中的第feature列中小于等于value的值所在的行数  从dataSet中取出对应的行（第一个）
def binSplitDataSet(dataSet, feature, value):
    # print("-----")
    # print(dataSet[:, feature])
    # print(dataSet[:, feature] > value)
    # print(np.nonzero(dataSet[:, feature] > value))
    # print(np.nonzero(dataSet[:, feature] > value)[0])
    # print(dataSet[np.nonzero(dataSet[:, feature] > value)[0], :])
    # print(np.nonzero(dataSet[:, feature] <= value)[0])
    # print("-----")

    # dataSet[:, feature]           dataSet 的第feature列
    # dataSet[:, feature] > value   将第feature列每个元素跟value对比，结果是个true/false的list, 例如[[False] \n [ True] \n [False] \n [False]]
    # np.nonzero(dataSet[:, feature] > value)       将上述结果的非零元素提取出来
    # np.nonzero(dataSet[:, feature] > value)[0]    满足 > value的值的所在行值
    # dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]        按照行值从dataSet中取出该行
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :][0]        # 满足
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :][0]

    return mat0, mat1


# 创建叶节点的函数
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1])  # dataSet最后一行的均值


# 误差计算函数：方差  *  行数
def regErr(dataSet):
    # dataSet最后一列的方差  *  行数
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]    # var 方差, np.shape(dataSet)[0]   dataSet的行数


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]
    tolN = ops[1]
    # if all the target variables are the same value: quit and return value
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # exit cond 1
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:, featIndex]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS

    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue
    # returns the best feature to split on
    # and the value used for that split


# 创建树
#   dataSet     原始数据
#   leafType    叶节点函数
#   errType     误差计算函数
#   ops
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat is None:
        return val

    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)

    return retTree


def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))   # 生成 m * n 的矩阵，所有的值都是1
    Y = np.mat(np.ones((m, 1)))   # 生成 m * 1 的矩阵，所有的值都是1

    X[:, 1: n] = dataSet[:, 0: n-1]     # 将dataSet的 0 ~ n-1 列覆盖 X的 1~n列的数值（X的第0列不变，仍然为1）
    Y = dataSet[:, -1]                  # 将dataSet的 最后一列 覆盖 Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\ try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)

    return ws, X, Y


def modelLeaf(dataSet):     # create linear model and return coeficients
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))

