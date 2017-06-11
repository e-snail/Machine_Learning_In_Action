
import numpy


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 1是带有侮辱性的，0不是
    classVec = [0, 1, 0, 1, 0, 1]

    return postingList, classVec


# 创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        # 操作符用于求两个集合的并集
        vocabSet = vocabSet | set(document)

    return list(vocabSet)


# 输入: vocabList 词汇表, inputSet 要分类的文档
# 输出: 文档词向量, 向量的每一元素为1或0, 分别表示词汇表中的单词在输入文档中是否出现
def setOfWords2Vec(vocabList, inputSet):

    # 函数首先创建一个和词汇表等长的向量,并将其元素都设置为0
    returnVec = [0] * len(vocabList)

    # 遍历文档中的所有单词, 如果出现了词汇表中的单词, 则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)

    return returnVec


# training module ??????????
def trainNB0(trainMatrix, trainCategory):
    # trainMatrix is [numTrainDocs, numWords]
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # sum(trainCategory)是所有侮辱性评论的个数

    # 生成值为1 x numWords的矩阵，初始值设为1
    p0Num = numpy.ones(numWords)  # numpy.ones(numWords)
    p1Num = numpy.ones(numWords)  # numpy.ones(numWords)   [1, 1, 1, ...] for numWords times
    p0Denom = 2.0
    p1Denom = 2.0                    # change to 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = numpy.log(p1Num/p1Denom)          # change to log()
    p0Vect = numpy.log(p0Num/p0Denom)          # change to log()

    return p0Vect, p1Vect, pAbusive


# ???
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # vec2Classify: 1 x N 矩阵
    # p1Vec: N x M 矩阵
    p1 = sum(vec2Classify * p1Vec) + numpy.log(pClass1)  # element-wise mult
    p0 = sum(vec2Classify * p0Vec) + numpy.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def testingNB():
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))

    p0V, p1V, pAb = trainNB0(numpy.array(trainMat), numpy.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))

    testEntry = ['stupid', 'garbage']
    thisDoc = numpy.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


# 输入大字符串
# 输出单词列表
def textParse(bigString):  # input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = [];
    classList = [];
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    trainingSet = range(50);
    testSet = []  # create test set
    for i in range(10):
        randIndex = int(numpy.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(numpy.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', float(errorCount) / len(testSet))
    # return vocabList,fullText


# 4.7 使用朴素贝叶斯分类器从个人广告中获取区域倾向

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []

    minLen = min(len(feed1['entries']), len(feed0['entries']))

    # 每次访问一条RSS源
    # docList 是一个[[xx, yy], [zz, yy], ...]
    # fullText 是 [xx, yy, zz, yy]
    # classList 是 [1, 0, 1, 0, 1, 0, ....]
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])        # 帖子正文的内容
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1

        # 解释
        # append 追加任意类型元素作为最后一个元素    li = ['a', 'b', 'c']  li.append(['d', 'e', 'f'])  => ['a', 'b', 'c', ['d', 'e', 'f']]
        # extend 扩展一个list类型的元素            li = ['a', 'b', 'c']  li.extend(['d', 'e', 'f'])   => li ['a', 'b', 'c', 'd', 'e', 'f']

        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 创建所有帖子的词表list
    vocabList = createVocabList(docList)  # create vocabulary
    # 找出出现频率最高的30个单词
    # ('and', 75), ('for', 39), ('the', 35), ('you', 34), ('looking', 26)...
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words

    # 删除这个30个单词
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = list(range(2 * minLen))

    testSet = []  # create test set

    for i in range(20):
        randIndex = int(numpy.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    # 结果类似
    # trainingSet  [1, 2, 6, 7, 8, 13, 15, 16, 17, 18, 20, 23, 24, 26, 27, 28, 30, 31, 32, 35, 36, 37, 38, 40, 41, 42, 43, 45, 47, 49]
    # testSet [19, 10, 14, 44, 25, 9, 0, 34, 4, 11, 46, 21, 12, 22, 5, 29, 39, 33, 3, 48]

    trainMat = []
    trainClasses = []

    # 训练集trainMat中加入 每个帖子的词表矢量
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 训练分类器
    p0V, p1V, pSpam = trainNB0(numpy.array(trainMat), numpy.array(trainClasses))
    print("pSpam")
    print(pSpam)  # 0.5代表什么？？？？

    # 从测试集中抽取帖子形成词表矢量，结果跟已知的分类作对比
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(numpy.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is: ', float(errorCount) / len(testSet))

    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0:
            topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0:
            topNY.append((vocabList[i], p1V[i]))

    # sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    # print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    # for item in sortedSF:
    #     print(item[0])
    #
    # sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    # print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    # for item in sortedNY:
    #     print(item[0])