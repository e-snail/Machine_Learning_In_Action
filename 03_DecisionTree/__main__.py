
# !/usr/bin/env python3

import trees

dataSet, labels = trees.createDataSet()

# print("dataSet=" + dataSet)

shannonEnt = trees.calcShannonEnt(dataSet)

print("shannonEnt=" + str(shannonEnt))

decission_tree = trees.createTree(dataSet, labels)

print("decission_tree:")
print(decission_tree)