
import regressiontree

import numpy as np

myDat = regressiontree.loadDataSet('data/ex00_bk.txt')

myMat = np.mat(myDat)

print(myMat)

regTree = regressiontree.createTree(myMat)

print(regTree)