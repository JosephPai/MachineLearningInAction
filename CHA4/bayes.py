from  numpy import  *

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

#统计所有文档中出现的词条列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)     # 求并集
    return list(vocabSet)

#根据词条列表中的词条是否在文档中出现(出现1，未出现0)，将文档转化为词条向量
def setOfWords2Vec(vocabList, inputSet):
    # 新建一个长度为vocabSet的列表，并且各维度元素初始化为0
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:   print("the word: %s is not in my Vocablary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

#训练算法，从词向量计算概率p(w0|ci)...及p(ci)
#@trainMatrix：由每篇文档的词条向量组成的文档矩阵
#@trainCategory:每篇文档的类标签组成的向量
def trainNBO(trainMatrix, trainCategory):
    # 获取文档矩阵中文档的数目
    numTrainDocs = len(trainMatrix)
    # 获取词条向量的长度
    numWords = len(trainMatrix[0])
    # 所有文档中属于类1所占的比例p(c=1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords);    p1Num = ones(numWords)
    p0Denom = 2.0;  p1Denom = 2.0
    # 遍历每一篇文档的词条向量
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 统计所有类别为1的词条向量中各个词条出现的次数
            p1Num += trainMatrix[i]
            # 统计类别为1的词条向量中出现的所有词条的总数
            # 即统计类1所有文档中出现单词的数目
            p1Denom += sum(trainMatrix[i])
        else:
            # 统计所有类别为0的词条向量中各个词条出现的次数
            p0Num += trainMatrix[i]
            # 统计类别为0的词条向量中出现的所有词条的总数
            # 即统计类0所有文档中出现单词的数目
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive

#朴素贝叶斯分类函数
#@vec2Classify:待测试分类的词条向量
#@p0Vec:类别0所有文档中各个词条出现的频数p(wi|c0)
#@p0Vec:类别1所有文档中各个词条出现的频数p(wi|c1)
#@pClass1:类别为1的文档占文档总数比例
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

#分类测试整体函数
def testingNB():
    # 由数据集获取文档矩阵和类标签向量
    listOPosts, listClasses = loadDataSet()
    # 统计所有文档中出现的词条，存入词条列表
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        # 将每篇文档利用words2Vec函数转为词条向量，存入文档矩阵中
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 将文档矩阵和类标签向量转为NumPy的数组形式，方便接下来的概率计算
    # 调用训练函数，得到相应概率值
    p0V, p1V, pAb = trainNBO(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as: ', classifyNB(thisDoc, p0V, p1V, pAb))


#贝叶斯算法实例：过滤垃圾邮件

#处理数据长字符串
#1 对长字符串进行分割，分隔符为除单词和数字之外的任意符号串
#2 将分割后的字符串中所有的大些字母变成小写lower(),并且只
#保留单词长度大于3的单词
def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]; classList=[];   fullText=[]
    for i in list(range(1,26)):
        # 打开并读取指定目录下的本文中的长字符串，并进行处理返回
        wordList = textParse(open('email/spam/%d.txt' % i,
                                  encoding='gb18030',errors='ignore').read())
        # 将得到的字符串列表添加到docList
        docList.append(wordList)
        # 将字符串列表中的元素添加到fullTest
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i,
                                  encoding='gb18030',errors='ignore').read())
        docList.append(wordList)
        fullText.append(wordList)
        classList.append(0)
    # 将所有邮件中出现的字符串构建成字符串列表
    vocabList = createVocabList(docList)
    trainingSet = list(range(50)); testSet=[]
    # 随机选取1~50中的10个数，作为索引，构建测试集
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将选出的数的列表索引值添加到testSet列表中
        testSet.append(trainingSet[randIndex])
        # 从整数列表中删除选出的数，防止下次再次选出
        # 同时将剩下的作为训练集
        del(trainingSet[randIndex])
    trainMat = [];  trainClasses=[]
    # 遍历训练集中的每个字符串列表
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print("the error rate is:", float(errorCount)/len(testSet))

def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1),\
                        reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    import feedparser
    docList=[]; classList=[]; fullText=[]
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:   vocabList.remove(pairW[0])
    trainingSet = list(range(2*minLen)); testSet=[]
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat=[]; trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNBO(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != \
            classList[docIndex]:
            errorCount += 1
    print('The error rate is:', float(errorCount)/len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny,sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY=[];   topSF=[]
    for i in range(len(p0V)):
        if p0V[i] > -6.0:   topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -6.0:   topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print("SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**")
    for item in sortedSF:
        print(item[0])
    sortedNY = sorted(topNY, key=lambda  pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sortedNY:
        print(item[0])