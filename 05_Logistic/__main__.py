
import logRegres
import numpy as nm

dataArr, labelMat = logRegres.loadDataSet()

weights = logRegres.gradAscent(dataArr, labelMat)

print(weights)

# x = nm.arange(-3.0, 3.0, 0.1)
# y = (-weights[0] - weights[1] * x) / weights[2]
#
# print(y)
