
from math import log
import operator


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']

    return dataSet, labels


# 计算dataSet的熵值
def calcShannonEnt(dataSet):
    # dataSet行数
    numEntries = len(dataSet)

    # labelCounts字典，存储所有分类(yes/no)的个数
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        # print("currentLabel=" + str(currentLabel))
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

    shannonEnt = 0.0

    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        # log base 2
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt


# dataSet   数据集
# axis      列序号
# value     数值
# 从数据集dataSet拆分出一个新的数据集，条件是axis列的值都等于value
def splitDataSet(dataSet, axis, value):

    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]             # featVec[:axis]结果是空
            reducedFeatVec.extend(featVec[axis+1:])     # extend 列插入：[axis+1:]表示axis+1列及其后所有的数据
            retDataSet.append(reducedFeatVec)           # append 行插入：往retDataSet插入一行

    return retDataSet


# 选择最好的数据集划分方法
# input: 原始数据集
# output: 原始数据集的列序号（以该列的特征值划分数据集最有效）
def chooseBestFeatureToSplit(dataSet):
    # 特征个数
    numFeatures = len(dataSet[0]) - 1      # the last column is used for the labels
    # 初始熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 记录最好的feature index
    bestFeature = -1

    # 对所有特征进行迭代
    for i in range(numFeatures):
        # dataSet中第i列的所有特征值
        featList = [example[i] for example in dataSet]  # create a list of all the examples of this feature

        # 特征值去重
        uniqueVals = set(featList)                      # get a set of unique values

        newEntropy = 0.0

        for value in uniqueVals:
            # 从dataSet中抽取子数据集subDataSet，条件是：第i列特征值中等于value的
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            # 对所有"唯一特征值"得到的熵求和
            newEntropy += prob * calcShannonEnt(subDataSet)

        # 计算信息增益 = 原始数据集的熵 - 基于（每列的）唯一特征值子数据集的熵（和值）
        infoGain = baseEntropy - newEntropy     # calculate the info gain; ie reduction in entropy

        # 记录信息增益最大值和特征列值
        if infoGain > bestInfoGain:       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i

    # 返回最佳特征值的列序号
    return bestFeature                      # returns an integer


# classList分类列表
# output: 列表中出现次数最多的分类
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0

        classCount[vote] += 1

    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


def createTree(dataSet, labels):

    # 以最后一列（分类）建立list  classList（有重复值）
    classList = [example[-1] for example in dataSet]

    # 递归终止条件1：
    # classList中第一个类别的个数等于classList的总条数，说明只有一个分类，停止划分
    # 返回唯一的分类
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 递归终止条件2：
    # 数据集只有一列时（遍历完所有特征后），停止划分，返回出现次数最多的分类
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择能得出最佳划分的特征值（序列值等于bestFeat）
    bestFeat = chooseBestFeatureToSplit(dataSet)

    # 保存每次筛选出来的特征值名bestFeatLabel到myTree中
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}

    # 从labels中删除i序列的值
    del(labels[bestFeat])

    # 原始数据集中bestFeat列的所有数据，然后去重得到uniqueVals
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)

    # 根据bestFeat列的所有特征值划划分数据集，并递归调用createTree函数，直到达到终止条件
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    # 决策树表达式
    return myTree