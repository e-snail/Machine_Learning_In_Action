
import numpy as nm
import adaboost

dataMat, classLabels = adaboost.loadSimpData()

D = nm.mat(nm.ones((5, 1)) / 5)

print(D)

bestStump, minError, bestClasEst = adaboost.buildStump(dataMat, classLabels, D)

print("bestStump->")
print(bestStump)
print("minError->")
print(minError)
print("bestClasEst->")
print(bestClasEst)
