from numpy import *
from math import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = [];   labelMat=[]
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#标准线性回归算法
#ws=(X.T*X).I*(X.T*Y)
def standRegres(xArr, yArr):
    xMat = mat(xArr);   yMat = mat(yArr).T
    # 求矩阵的内积
    xTx = xMat.T*xMat
    # numpy线性代数库linalg
    # 调用linalg.det()计算矩阵行列式
    # 计算矩阵行列式是否为0
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    # 如果可逆，根据公式计算回归系数
    ws = xTx.I * (xMat.T*yMat)
    # 可以用yHat=xMat*ws计算实际值y的预测值
    # 返回归系数
    return ws

#局部加权线性回归

#每个测试点赋予权重系数
#@testPoint:测试点
#@xArr：样本数据矩阵
#@yArr：样本对应的原始值
#@k：用户定义的参数，决定权重的大小，默认1.0
def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = mat(xArr); yMat=mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m))) # 对角矩阵
    for j in range(m):#循环遍历各个样本
        diffMat = testPoint - xMat[j,:] #计算预测点与该样本的偏差
        weights[j,j] = exp(diffMat*diffMat.T/(-2.0*k**2))  #根据偏差利用高斯核函数赋予该样本相应的权重
    # 将权重矩阵应用到公式中
    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat.T * (weights*yMat))
    return testPoint*ws

#测试集进行预测
def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#计算平方误差的和
def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

#岭回归
#@xMat:样本数据
#@yMat：样本对应的原始值
#@lam：惩罚项系数lamda，默认值为0.2
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    # 添加惩罚项，使矩阵xTx变换后可逆
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

def ridgeTest(xArr, yArr):
    # 标准化处理
    xMat = mat(xArr);
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean  # to eliminate X0 take mean off of Y
    # regularize X's
    xMeans = mean(xMat, 0)  # calc mean then subtract it off
    xVar = var(xMat, 0)  # calc variance of Xi then divide by it
    # 特征-均值/方差
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30 #在30个不同的lamda下进行测试
    wMat = zeros((numTestPts, shape(xMat)[1]))  #30次的结果保存在wMat中
    for i in range(numTestPts):
        # 计算对应lamda回归系数，lamda以指数形式变换
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T
    return wMat

def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

#前向逐步回归( 贪心算法
def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = mat(xArr);   yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat-yMean
    xMat = regularize(xMat)
    m,n = shape(xMat)
    # 将每次迭代中得到的回归系数存入矩阵
    returnMat = zeros((numIt,n))
    ws = zeros((n,1));  wsTest = ws.copy(); wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)
        lowestError = inf;
        for j in range(n):
            # 对每个特征的系数执行增加和减少eps*sign操作
            for sign in [-1,1]:
                wsTest = ws.copy()
                wsTest[j] += eps*sign
                # 变化后计算相应预测值
                yTest = xMat*wsTest
                # 保存最小的误差以及对应的回归系数
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i,:] = ws.T
    return returnMat


from time import sleep
import json
from urllib.request import urlopen
#@retX:样本玩具特征矩阵
#@retY：样本玩具的真实价格
#@setNum：获取样本的数量
#@yr：样本玩具的年份
#@numPce:玩具套装的零件数
#@origPce:原始价格
def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    sleep(10)
    myAPIstr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
    searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
    pg = urlopen(searchURL)
    retDict = json.loads(pg.read())
    # 遍历数据的每一个条目
    for i in range(len(retDict['items'])):
        try:
            currItem = retDict['items'][i]#获得当前条目
            if currItem['product']['condition'] == 'new':#当前条目对应的产品为新产品
                newFlag = 1
            else: newFlag = 0
            listOfInv = currItem['product']['inventories']#得到当前目录产品的库存列表
            for item in listOfInv:#遍历库存中的每一个条目
                sellingPrice = item['price']#得到该条目玩具商品的价格
                if  sellingPrice > origPrc * 0.5:#价格低于原价的50%视为不完整套装
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])#将符合条件套装信息作为特征存入数据矩阵
                    retY.append(sellingPrice)#将对应套装的出售价格存入矩阵
        except: print('problem with item %d' % i)

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

#训练算法：建立模型
#交叉验证测试岭回归
#@xArr:从网站中获得的玩具套装样本数据
#@yArr：样本对应的出售价格
#@numVal:交叉验证次数
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = list(range(m))
    # 将每个回归系数对应的误差存入矩阵
    errorMat = zeros((numVal,30))
    for i in range(numVal):
        trainX=[];  trainY=[]
        testX=[];   testY=[]
        random.shuffle(indexList)
        for j in range(m):
            if j<m*0.9: #数据集90%作为训练集
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeTest(trainX,trainY) #利用训练集计算岭回归系数
        for k in range(30):
            matTestX = mat(testX);  matTrainX=mat(trainX)
            meanTrain = mean(matTrainX,0)   #求训练集特征的均值
            varTrain = var(matTrainX,0) #计算训练集特征的方差
            matTestX = (matTestX-meanTrain)/varTrain    #测试集用与训练集相同的参数进行标准化
            yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#对每组回归系数计算测试集的预测值
            errorMat[i,k] = rssError(yEst.T.A, array(testY))    #将原始值和预测值的误差保存
    meanErrors = mean(errorMat,0)   #对误差矩阵中每个lamda对应的10次交叉验证的误差结果求均值
    minMean = float(min(meanErrors))    #找到最小的均值误差
    # 将均值误差最小的lamda对应的回归系数作为最佳回归系数
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    xMat = mat(xArr);   yMat = mat(yArr).T
    meanX = mean(xMat,0);   varX=var(xMat,0)
    unReg = bestWeights/varX
    print("the best model from Ridge Regression is :\n", unReg)
    print("with constant term: ", \
          -1*sum(multiply(meanX,unReg)) + mean(yMat))

