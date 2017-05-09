
import bayes_algorithm
import feedparser

postingList, classVec = bayes_algorithm.loadDataSet()

# 单词列表，不含重复单词
myVocabList = bayes_algorithm.createVocabList(postingList)

# print(myVocabList)

worthlessIndex = 0
stupidIndex = 0
helpIndex = 0
isIndex = 0
dogIndex = 0

for i in range(len(myVocabList)):
    if myVocabList[i] is "stupid":
        stupidIndex = i
    elif myVocabList[i] is "worthless":
        worthlessIndex = i
    elif myVocabList[i] is "help":
        helpIndex = i
    elif myVocabList[i] is "is":
        isIndex = i
    elif myVocabList[i] is "dog":
        dogIndex = i

#
trainMat = []
for postInDoc in postingList:
    trainMat.append(bayes_algorithm.setOfWords2Vec(myVocabList, postInDoc))

p0V, p1V, pAb = bayes_algorithm.trainNB0(trainMat, classVec)

bayes_algorithm.testingNB()


# --------------------------------------------------------------------------------- #

print("\n4.7 使用朴素贝叶斯分类器从个人广告中获取区域倾向\n")

ny = feedparser.parse('https://newyork.craigslist.org/search/stp?format=rss')
sf = feedparser.parse('https://sfbay.craigslist.org/search/stp?format=rss')

vocabList, pSF, pNY = bayes_algorithm.localWords(ny, sf)

vocabList, pSF, pNY = bayes_algorithm.localWords(ny, sf)

bayes_algorithm.getTopWords(ny, sf)


