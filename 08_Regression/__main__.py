
import numpy as np
import regression

import matplotlib.pyplot as plt

dataMat, labelMat = regression.loadDataSet('ex0.txt')

# -------------------------------------------------
# 标准回归法计算预测值

ws = regression.standRegression(dataMat, labelMat)

print("回归系数")
print(ws)

xMat = np.mat(dataMat)      # 实际输入值
yMat = np.mat(labelMat)     # 实际输出值

yHat = xMat * ws    # 预测的输出值

relation = np.corrcoef(yHat.T, yMat)

print("实际输出和预测输出的相关性")
print(relation)

# 绘制拟合线
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
#
# xCopy = xMat.copy()
# xCopy.sort(0)
# yHat = xCopy * ws
#
# ax.plot(xCopy[:, 1], yHat)
#
# plt.show()


# -------------------------------------------------
# 局部加权回归法计算预测值

xArr, yArr = regression.loadDataSet('ex0.txt')
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.01)

xMat = np.mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c='red')
plt.show()

