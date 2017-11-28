from math import *
from numpy import *
def loadDataSet():
    dataMat = [];   labelMat=[]
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return longfloat(1.0/(1+exp(-inX))) # 使用longfloat防止溢出

# 梯度上升
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat(dataMatIn) #每列分别代表每个不同的特征，每行则代表每个训练样本
    labelMat = mat(classLabels).transpose() # 转换为numpy矩阵格式
    m,n=shape(dataMatIn)
    alpha = 0.001
    maxCycles = 500 #迭代次数
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

# 随机梯度上升（大数据量
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

# 随机梯度上升改进
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01  # 每次调整步长
            randIndex = int(random.uniform(0, len(dataMatrix)))  # 随机选取
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (list(dataMatrix)[randIndex])
    return weights

# 给定训练好的weights，画出图线
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = [];    ycord1=[]
    xcord2 = [];    ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');   plt.ylabel('X2')
    plt.show()

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:    return 1.0
    else:   return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];   trainLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainLabels, 500)
    errorCount = 0; numTestVec=0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)!=int(currLine[21])):
            errorCount+=1
    errorRate = (float(errorCount)/numTestVec)
    print("The error rate of this test is:%f" % errorRate)
    return errorRate

def multiTest():
    numTests=10;    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("After %d iteration the average error rate is: %f" % \
          (numTests, errorSum/float(numTests)))