from numpy import *
def loadSimpData():
    datMat = matrix([[1.0, 2.1],
                    [2.0, 1.1],
                    [1.3, 1.0],
                    [1.0,1.0],
                    [2.0,1.0]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = [];   labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#构建单层分类器
#单层分类器是基于最小加权分类错误率的树桩
#伪代码
#将最小错误率minError设为+∞
#对数据集中的每个特征(第一层特征)：
    #对每个步长(第二层特征)：
        #对每个不等号(第三层特征)：
            #建立一颗单层决策树并利用加权数据集对它进行测试
            #如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
#返回最佳单层决策树

#单层决策树的阈值过滤函数
def stumpClassify(dataMatix, dimen, threshVal, threshIneq):
    # 对数据集每一列的各个特征进行阈值过滤
    retArray = ones((shape(dataMatix)[0],1))
    # 阈值的模式，将小于某一阈值的特征归类为-1
    if threshIneq == 'lt':
        retArray[dataMatix[:,dimen] <= threshVal] = -1.0
    # 将大于某一阈值的特征归类为-1
    else:
        retArray[dataMatix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    # 将数据集和标签列表转为矩阵形式
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    # 步长或区间总数    最优决策树信息  最优单层决策树预测结果
    numSteps = 10.0;    bestStump={};   bestClasEst=mat(zeros((m,1)))
    # 最小错误率初始化为+∞
    minError = inf
    # 遍历每一列的特征值
    for i in range(n):
        # 找出列中特征值的最小值和最大值
        rangeMin = dataMatrix[:,i].min();   rangeMax=dataMatrix[:,i].max()
        # 求取步长大小或者说区间间隔
        stepSize = (rangeMax-rangeMin)/numSteps
        # 遍历各个步长区间
        for j in range(-1, int(numSteps)+1):
            # 两种阈值过滤模式
            for inequal in ['lt', 'gt']:
                # 阈值计算公式：最小值+j(-1<=j<=numSteps+1)*步长
                threshVal = (rangeMin+float(j)*stepSize)
                # 选定阈值后，调用阈值过滤函数分类预测
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                # 初始化错误向量
                errArr = mat(ones((m,1)))
                # 将错误向量中分类正确项置0
                errArr[predictedVals == labelMat] = 0
                # 计算"加权"的错误率
                weightedError = D.T * errArr
                # print("split: dim %d, thresh %.2f, thresh inequal:\ "
                #       "%s, the weighted error is %.3f" % \
                #       (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim']=i
                    bestStump['thresh']=threshVal
                    bestStump['ineq']=inequal
    # 返回最佳单层决策树相关信息的字典，最小错误率，决策树预测输出结果
    return bestStump, minError, bestClasEst


#完整AdaBoost算法实现
#算法实现伪代码
#对每次迭代：
    #利用buildStump()函数找到最佳的单层决策树
    #将最佳单层决策树加入到单层决策树数组
    #计算alpha
    #计算新的权重向量D
    #更新累计类别估计值
    #如果错误率为等于0.0，退出循环

#adaBoost算法
#@dataArr：数据矩阵
#@classLabels:标签向量
#@numIt:迭代次数
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    # 弱分类器相关信息列表
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)  # 每个数据点的权重
    aggClassEst = mat(zeros((m,1))) # 每个数据点的类别估计累计值
    for i in range(numIt):
        # 根据当前数据集，标签及权重建立最佳单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        # 求单层决策树的系数alpha
        alpha = float(0.5 * log((1.0-error)/max(error, 1e-16)))
        # 存储决策树的系数alpha到字典
        bestStump['alpha'] = alpha
        # 将该决策树存入列表
        weakClassArr.append(bestStump)
        print("classEst:",classEst.T)
        # 预测正确为exp(-alpha),预测错误为exp(alpha)
        # 即增大分类错误样本的权重，减少分类正确的数据点权重
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        # 更新权值向量
        D = multiply(D, exp(expon))
        D = D/D.sum()
        # 累加当前单层决策树的加权预测值
        aggClassEst += alpha*classEst
        print("aggClassEst:", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T, ones((m,1)))
        errorRate = aggErrors.sum() / m
        print("total error:", errorRate, "\n")
        # 错误率为0.0退出循环
        if errorRate==0.0:break
    return weakClassArr, aggClassEst

#测试adaBoost，adaBoost分类函数
#@datToClass:测试数据点
#@classifierArr：构建好的最终分类器
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        # 每一个弱分类器对测试数据进行预测分类
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'],\
                                 classifierArr[i]['thresh'],\
                                 classifierArr[i]['ineq'])
        # 对各个分类器的预测结果进行加权累加
        aggClassEst += classifierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)

def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels)==1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndices = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndices.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0;   delY = yStep
        else:
            delX = xStep;   delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Postive Rate');   plt.ylabel('True Postive Rate')
    plt.title('ROC curve for Adaboost Horse Colic Detection System')
    ax.axis([0,1,1,1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)

#训练和测试分类器
def classify():
    #利用训练集训练分类器
    datArr,labelArr=loadDataSet('horseColicTraining2.txt')
    #得到训练好的分类器
    classifierArray=adaBoostTrainDS(datArr,labelArr,10)
    #利用测试集测试分类器的分类效果
    testArr,testLabelArr=loadDataSet('horseClicTest2.txt')
    prediction=adaClassify(testArr,classifierArray)
    #输出错误率
    num=shape(mat(labelArr))[1]
    errArr=mat(ones((num,1)))
    error=errArr[prediction!=mat(testLabelArr).T].sum()
    print("the errorRate is: %.2f",errorRate=float(error)/float((num)))