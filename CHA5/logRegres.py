from math import *
from numpy import *
#预处理数据
def loadDataSet():
    # 创建两个列表
    dataMat = [];   labelMat=[]
    # 打开文本数据集
    fr = open('testSet.txt')
    # 遍历文本的每一行
    for line in fr.readlines():
        # 对当前行去除首尾空格，并按空格进行分离
        lineArr = line.strip().split()
        # 将每一行的两个特征x1，x2，加上x0=1,组成列表并添加到数据集列表中
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        # 将当前行标签添加到标签列表
        labelMat.append(int(lineArr[2]))
    # 返回数据列表，标签列表
    return dataMat, labelMat

def sigmoid(inX):
    return longfloat(1.0/(1+exp(-inX))) # 使用longfloat防止溢出

# 梯度上升
#@dataMatIn：数据集
#@classLabels：数据标签
def gradAscent(dataMatIn, classLabels):
    # 将数据集列表转为Numpy矩阵
    dataMatrix = mat(dataMatIn) #每列分别代表每个不同的特征，每行则代表每个训练样本
    labelMat = mat(classLabels).transpose() # 转换为numpy矩阵格式
    m,n=shape(dataMatIn)
    alpha = 0.001   #学习步长
    maxCycles = 500 #迭代次数
    weights = ones((n,1))   #初始化权值参数向量每个维度均为1.0
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights) #求当前的sigmoid函数预测概率
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error    #更新权值参数
    return weights

# 随机梯度上升（大数据量
def stocGradAscent0(dataMatrix, classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    # 循环m次，每次选取数据集一个样本更新参数
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i]-h
        weights = weights + alpha*error*dataMatrix[i]
    return weights

# 随机梯度上升改进
#@dataMatrix：数据集列表
#@classLabels：标签列表
#@numIter：迭代次数，默认150
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):#循环每次迭代次数
        for i in range(m):#遍历行列表
            alpha = 4 / (1.0 + j + i) + 0.01  # 每次调整步长
            randIndex = int(random.uniform(0, len(dataMatrix)))  # 随机选取
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            # 选取该样本后，将该样本下标删除，确保每次迭代时只使用一次
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

#------------------------------实例：从疝气病预测病马的死亡率----------------------------
#1 准备数据：处理数据的缺失值
#这里将特征的缺失值补0，从而在更新时不影响系数的值
#2 分类决策函数
def classifyVector(inX, weights):
    # 计算logistic回归预测概率
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:    return 1.0
    else:   return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = [];   trainLabels = []
    for line in frTrain.readlines():#读取训练集文档的每一行
        currLine = line.strip().split('\t')#对当前行进行特征分割
        lineArr=[]#新建列表存储每个样本的特征向量
        for i in range(21):
            lineArr.append(float(currLine[i]))#将该样本的特征存入lineArr列表
        trainingSet.append(lineArr)#将该样本标签存入标签列表
        trainLabels.append(float(currLine[21]))#将该样本的特征向量添加到数据集列表
    # 调用随机梯度上升法更新logistic回归的权值参数
    trainWeights = stocGradAscent1(array(trainingSet), trainLabels, 500)
    errorCount = 0; numTestVec=0.0
    # 遍历测试数据集的每个样本
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
    # 设置测试次数为10次，并统计错误率总和
    numTests=10;    errorSum=0.0
    # 每一次测试算法并统计错误率
    for k in range(numTests):
        errorSum += colicTest()
    # 打印出测试10次预测错误率平均值
    print("After %d iteration the average error rate is: %f" % \
          (numTests, errorSum/float(numTests)))