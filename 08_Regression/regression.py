
import numpy as np

# ex0.txt 数据说明
# 第一行 1.000000	0.067732	3.176513
# 该点的坐标是[0.067732, 3.176513]   用x坐标值作为输入值，y坐标值作为真实输出值
# 回归方程有两个系数，所以输入值必须有两个，使用1.0000作为第一个输入值，可以假定偏移量就是一个常数。x作为第二个输出值。


# 从文件中读取数据
# 输出
#   dataMat   实际的输入数据
#   labelMat  实际的输出数据
def loadDataSet(fileName):
    # 取出每行数据的列数
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


# 输入：
#   xArr 输入值
#   yArr 实际输出值
# 输出：
#   回顾系数
# 使用标准回归函数计算回归系数
def standRegression(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat     # xMat.T是xMat的转置

    # numpy.linalg.det 用来计算举着xMat的行列式，行列式非0就说明矩阵有逆矩阵
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return

    # xTx.I 是xTx的逆矩阵
    ws = xTx.I * (xMat.T * yMat)    # 也等于 linalg.solve(xTx, xMat.T * yMat)
    # solve的说明  https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
    # solve用于解线性方程组AX==B，X = solve(A, B)
    # 这里的等式是： xTx.I * X = xMat.T * yMat    求解X

    return ws


# 局部加权线性回归
# 输入
#   testPoint 测试点
#   xArr    输入数据
#   yArr    实际输出数据
#   k       高斯核的参数
# 输出
#   testPoint的预测输出
#
# 计算过程
# 1 对每个输入点计算一个权重矩阵，该矩阵只包含对角元素，表示对角线上的点对输入点（计算输入点的预测值）的权重
# 2 计算权重矩阵使用的是高斯核
# 3 输入点 x 权重矩阵 = 预测值
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye((m)))
    # 遍历所有数据点，计算testPoint对每个数据点的权重
    for j in range(m):
        diffMat = testPoint - xMat[j, :]     #
        weights[j, j] = np.exp(diffMat*diffMat.T/(-2.0*k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws


# 循环调用lwlr计算所有点的预测值，存储到预测值矩阵中
# 输入
#   testArr 测试矩阵
#   xArr  测试矩阵的输入值
#   yArr  测试矩阵的实际输出值
#   k 高斯核系数
def lwlrTest(testArr, xArr, yArr, k=1.0):  #loops over all the data points and applies lwlr to each one
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

