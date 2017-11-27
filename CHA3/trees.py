from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

#得到熵之后，我们就可以按照获取最大信息增益的方法划分数据集，
#那么如何划分数据集以及如何度量信息增益
''''' 
现在这说一个append和extend的区别 
>>> a = [1,2,3] 
>>> b = [4,5,6] 
>>> a.append(b) 
>>> a 
[1, 2, 3, [4, 5, 6]] 
>>> a = [1,2,3] 
>>> a.extend(b) 
>>> a 
[1, 2, 3, 4, 5, 6] 
'''
#划分数据集：按照最优特征划分数据集
#@dataSet:待划分的数据集
#@axis:划分数据集的特征
#@value:特征的取值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选取特征，划分数据集，计算得出最好的划分数据集的特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       # 特征的个数
    baseEntropy = calcShannonEnt(dataSet)   #计算数据集的熵，放到baseEntropy中
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:    #下面是计算每种划分方式的信息熵,特征i个，每个特征value个值
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy #计算i个特征的信息熵
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return  bestFeature

#返回出现次数最多的分类名称
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),\
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]    #将最后一行的数据放到classList中
    if classList.count(classList[0]) == len(classList):
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return  myTree

#完成决策树的构造后，采用决策树实现具体应用
#@intputTree 构建好的决策树
#@featLabels 特征标签列表
#@testVec 测试实例
def classify(inputTree, featLabels ,testVec):
    # 找到树的第一个分类特征，或者说根节点'no surfacing'
    # 注意python2.x和3.x区别，2.x可写成firstStr=inputTree.keys()[0]
    # 而不支持3.x
    firstStr = list(inputTree.keys())[0]
    # 从树中得到该分类特征的分支，有0和1
    secondDict = inputTree[firstStr]
    # 根据分类特征的索引找到对应的标称型数据值
    # 'no surfacing'对应的索引为0
    featIndex = featLabels.index(firstStr)
    # 遍历分类特征所有的取值
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            # type()函数判断该子节点是否为字典类型
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            # 如果是叶子节点，则返回节点取值
            else:   classLabel = secondDict[key]
    return classLabel

#决策树的存储：python的pickle模块序列化决策树对象，使决策树保存在磁盘中
#在需要时读取即可，数据集很大时，可以节省构造树的时间
#pickle模块存储决策树
def storeTree(inputTree,filename):
    #导入pickle模块
    import pickle
    #创建一个可以'写'的文本文件
    #这里，如果按树中写的'w',将会报错write() argument must be str,not bytes
    #所以这里改为二进制写入'wb'
    fw=open(filename,'wb')
    #pickle的dump函数将决策树写入文件中
    pickle.dump(inputTree,fw)
    #写完成后关闭文件
    fw.close()
#取决策树操作
def grabTree(filename):
    import pickle
    #对应于二进制方式写入数据，'rb'采用二进制形式读出数据
    fr=open(filename,'rb')
    return pickle.load(fr)