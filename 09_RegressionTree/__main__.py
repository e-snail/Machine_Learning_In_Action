
import regressiontree

import numpy as np
import matplotlib.pyplot as plt

# show ex00.txt
myDat = regressiontree.loadDataSet('data/ex00.txt')
myMat = np.mat(myDat)
# print(myMat)
regTree = regressiontree.createTree(myMat)
print(regTree)

plt.plot(myMat[:, 0], myMat[:, 1], 'ro')
plt.show()

# show ex0.txt
myMat1 = regressiontree.loadDataSet('data/ex0.txt')
myMat1 = np.mat(myMat1)
regTree1 = regressiontree.createTree(myMat1)
print(regTree1)

plt.plot(myMat1[:, 1], myMat1[:, 2], 'ro')
plt.show()

