
# from numpy import *
import numpy as np
import operator
from os import listdir


def create_data_set():
    group = np.array([[1.0, 1.1],
                      [1.0, 1.0],
                      [0,   0],
                      [0,   0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


# k-近邻算法
# inX       用于分类的输入向量
# dataSet   输入的训练样本集
# labels    标签向量
# K         最近邻近邻居的数目
def classify0(inX, dataSet, labels, K):

    # Step1: 使用欧氏距离公式计算inX和dataSet的距离

    # array 的长度
    dataSetSize = dataSet.shape[0]
    # inX跟dataSet的差
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    # idffMat取平方
    sqDiffMat = diffMat ** 2
    # 差值相加
    sqDistances = sqDiffMat.sum(axis=1)
    # sqDistances值开平方
    distances = sqDistances ** 0.5

    # Step2: 选择距离最小的K个点

    # 按照相关度排序，值越小相关度越大
    sortedDistIndicies = distances.argsort()

    # 统计前K个标签中每个标签出现的次数
    classCount = {}
    for i in range(K):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1

    # Step3: 排序

    # 按照标签出现的次数排序，正序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]


# 将文本转换为NumPy
def file2matrix(filename):
    fr = open(filename)
    # get the number of lines in the file
    numberOfLines = len(fr.readlines())
    # prepare matrix to return
    returnMat = np.zeros((numberOfLines, 3))
    # prepare labels return
    classLabelVector = []

    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        # 索引值 -1表示列表中的最后一列元素, 利用这种负索引, 我们可以很方便地将列表的最后一列存储到向量classLabelVector中。
        # 必须明确地通知解释器,告 诉它列表中存储的元素值为整型,否则Python语言会将这些元素当作字符串处理
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    # 返回数据矩阵和分类标签
    return returnMat, classLabelVector


# 规约化特征值，否则某些特征的值过大会影响结果
#   下面的公式可以将任意取值范围的特征值转化为0到1区间内的值:
#   newValue = (oldValue - min)/(max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals

    # dataSet的长度
    shape = np.shape(dataSet)

    # 返回来一个给定形状和类型的用0填充的数组；
    # 用法：zeros(shape, dtype=float, order='C')
    #   shape:形状
    #   dtype:数据类型，可选参数，默认numpy.float64
    #   order:可选参数，c代表行优先；F代表列优先
    normDataSet = np.zeros(shape)

    m = dataSet.shape[0]

    # tile的用法
    #   numpy.tile([0,0],(1,1))#在列方向上重复[0,0]1次，行1次
    #   得到：array([[0, 0]])
    #
    #   tile(minVals, (m, 1))
    #   将minVals重复一次，行m次
    normDataSet = dataSet - np.tile(minVals, (m, 1))    # --->矩阵是原数据减去最小值
    normDataSet = normDataSet/np.tile(ranges, (m, 1))   # --->以上差值除以ranges

    # 返回
    #   normDataSet 规约后的数组
    #   最大和最小的差
    #   最小值
    return normDataSet, ranges, minVals


# 测试分类效果
def datingClassTest():
    hoRatio = 0.10      # hold out 10%
    # 创建数据集
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    # 规约数据集
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        # 使用分类器classfy0对测试数据进行评测
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))

        if (classifierResult != datingLabels[i]):
            errorCount += 1.0

    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(errorCount)


# 将文本转换成1x1024的向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)

    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect


# 手写数字测试
def handwritingClassTest():
    hwLabels = []
    # 加载训练集数据
    trainingFileList = listdir('digits/trainingDigits')           #load the training set

    # 生成一个{m, 1024}的矩阵, m是训练集的文件个数
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)

    # 对测试数据应用分类算法
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        # print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))

        if (classifierResult != classNumStr):
            errorCount += 1.0

    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))