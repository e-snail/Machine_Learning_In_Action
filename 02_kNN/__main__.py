
# !/usr/bin/env python3

import kNN
import matplotlib
# import matplotlib.pyplot as plt

# group, labels = kNN.create_data_set();
# print(kNN.classify0([0,0], group, labels, 3))

# datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:1], datingDataMat[:2])
# plt.show()

print("预测约会对象的魅力值")
kNN.datingClassTest()

print("识别手写字母")
kNN.handwritingClassTest()