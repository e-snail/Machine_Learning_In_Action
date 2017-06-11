
import numpy as nm


def loadSimpData():
    datMat = nm.matrix\
        ([[1.,  2.1],
        [2.,  1.1],
        [1.3,  1.],
        [1.,  1.],
        [2.,  1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels


def loadDataSet(fileName):      #general function to parse tab -delimited floats
    numFeat = len(open(fileName).readline().split('\t')) # get number of fields
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# just classify the data
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = nm.ones((nm.shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    print("dataArr->")
    print(dataArr)

    dataMatrix = nm.mat(dataArr)
    print("dataMatrix->")
    print(dataMatrix)

    # 将classLabels矩阵转置
    labelMat = nm.mat(classLabels).T
    print("labelMat->")
    print(nm.mat(classLabels))
    print(labelMat)

    m, n = nm.shape(dataMatrix)
    print("m->")
    print(m)
    print("n->")
    print(n)

    numSteps = 10.0
    bestStump = {}

    bestClasEst = nm.mat(nm.zeros((m, 1)))
    print("bestClasEst->")
    print(bestClasEst)

    # init error sum, to +infinity
    minError = nm.inf

    # 遍历矩阵所有 列 数据
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()  # dataMatrix[:, i]取矩阵dataMatrix的第i列, .min() 最小数  .max() 最大数
        rangeMax = dataMatrix[:, i].max()

        stepSize = (rangeMax-rangeMin)/numSteps
        # loop over all range in current dimension
        for j in range(-1, int(numSteps)+1):
            # go over less than and greater than
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # call stump classify with i, j, lessThan
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = nm.mat(nm.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # calc total error multiplied by D
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minError, bestClasEst