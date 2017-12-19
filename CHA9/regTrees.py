from numpy import *

#解析文本数据
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split()
        # 将每行数据映射为浮点数
        fltLine = list(map(float, curLine))   # 映射的方式
        dataMat.append(fltLine)
    fr.close()

#拆分数据集函数，二元拆分法
#@dataSet：待拆分的数据集
#@feature：作为拆分点的特征索引
#@value：特征的某一取值作为分割值
def binSplitDataSet(dataSet, feature, value):
    # 采用条件过滤的方法获取数据集每个样本目标特征的取值大于
    # value的样本存入mat0
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    # 同上，样本目标特征取值不大于value的样本存入mat1
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0, mat1

#叶节点生成函数
def regLeaf(dataSet):#returns the value used for each leaf
    return mean((dataSet[:,-1]))#数据集列表最后一列特征值的均值作为叶节点返回

#误差计算函数
def regErr(dataSet):
    # 计算数据集最后一列特征值的均方差 * 数据集样本数，得到总方差返回
    return var(dataSet[:,-1]) * shape(dataSet)[0]

#创建树函数
#@dataSet：数据集
#@leafType：生成叶节点的类型 1 回归树：叶节点为常数值 2 模型树：叶节点为线性模型
#@errType：计算误差的类型 1 回归错误类型：总方差=均方差*样本数
#                         2 模型错误类型：预测误差(y-yHat)平方的累加和
#@ops：用户指定的参数，包含tolS：容忍误差的降低程度 tolN：切分的最少样本数
def creatTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    # 选取最佳分割特征和特征值
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    if feat==None:  return val  #如果特征为none，直接返回叶节点值
    retTree = {}    #树的类型是字典类型
    retTree['spInd'] = feat #树字典的一个元素是切分的最佳特征
    retTree['spVal'] = val  #第二个元素是最佳特征对应的最佳切分特征值
    # 根据特征索引及特征值对数据集进行二元拆分，并返回拆分的两个数据子集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 第三个元素是树的左分支，通过lSet子集递归生成左子树
    retTree['left'] = creatTree(lSet, leafType, errType, ops)
    # 第四个元素是树的右分支，通过rSet子集递归生成右子树
    retTree['right'] = creatTree(rSet, leafType, errType, ops)
    return retTree  #返回生成的数字典

# 核心函数
#选择最佳切分特征和最佳特征取值函数
#@dataSet：数据集
#@leafType：生成叶节点的类型，默认为回归树类型
#@errType：计算误差的类型，默认为总方差类型
#@ops：用户指定的参数，默认tolS=1.0，tolN=4
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0];  tolN=ops[1]     # tolS是容许的误差下降值，tolN是切分的最少样本数
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:#数据集最后一列所有的值都相同
        # 最优特征返回none，将该数据集最后一列计算均值作为叶节点值返回
        return None, leafType(dataSet)
    m,n = shape(dataSet) #数据集的行与列
    S = errType(dataSet)#计算未切分前数据集的误差
    # 初始化最小误差；最佳切分特征索引；最佳切分特征值
    bestS = inf;      bestIndex=0;       bestValue=0
    # 遍历数据集所有的特征，除最后一列目标变量值
    for featIndex in range(n-1):
        # 遍历该特征的每一个可能取值
        for splitVal in set(dataSet(dataSet[:,featIndex])):
            # 以该特征，特征值作为参数对数据集进行切分为左右子集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果左分支子集样本数小于tolN或者右分支子集样本数小于tolN，跳出本次循环
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0]<tolN):    continue
            # 计算切分后的误差，即均方差和
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:    #保留最小误差及对应的特征及特征值
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if(S-bestS) < tolS:#如果切分后比切分前误差下降值未达到tolS
        return None, leafType(dataSet)#不需切分，直接返回目标变量均值作为叶节点
    # 检查最佳特征及特征值是否满足不切分条件
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if(shape(mat0)[0] < tolN) or (shape(mat1)[0]<tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征及最佳切分特征取值
    return bestIndex, bestValue

#根据目标数据的存储类型是否为字典型，是返回true，否则返回false
def isTree(obj):
    return (type(obj).__name__=='dict')

#获取均值函数
def getMean(tree):
    # 树字典的右分支为字典类型：递归获得右子树的均值
    if isTree(tree['right']):  tree['right']=getMean(tree['right'])
    # 树字典的左分支为字典类型：递归获得左子树的均值
    if isTree(tree['left']):  tree['left']=getMean(tree['left'])
    # 递归直至找到两个叶节点，求二者的均值返回
    return (tree['left'] + tree['right']) / 2.0

#剪枝函数
#@tree:树字典
#@testData:用于剪枝的测试集
def prune(tree, testData):
    # 测试集为空，直接对树相邻叶子结点进行求均值操作
    if(shape(testData)[0] == 0):    return getMean(tree)

    # 左右分支中有非叶子结点类型
    if (isTree(tree['right']) or isTree(tree['left'])):
        # 利用当前树的最佳切分点和特征值对测试集进行树构建过程
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 左分支非叶子结点，递归利用测试数据的左子集对做分支剪枝
    if isTree(tree['left']):    tree['left'] = prune(tree['left'],lSet)
    # 同理，右分支非叶子结点，递归利用测试数据的右子集对做分支剪枝
    if isTree(tree['right']):   tree['right'] = prune(tree['right'], rSet)

    # 左右分支都是叶节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        # 利用该子树对应的切分点对测试数据进行切分(树构建)
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 如果这两个叶节点不合并，计算误差，即（实际值-预测值）的平方和
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'], 2)) + \
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right']) / 2.0 #求两个叶结点值的均值
        # 如果两个叶节点合并，计算合并后误差,即(真实值-合并后值）平方和
        errorMerge = sum(power(testData[:,-1] - treeMean, 2 ))
        if errorMerge < errorNoMerge:   #合并后误差小于合并前误差
            print("merging")
            return treeMean #和并两个叶节点，返回合并后节点值
        else:   return tree #否则不合并，返回该子树
    else:   return tree #不合并，直接返回树




#模型树叶节点生成函数
def linearSolve(dataSet):
    m,n = shape(dataSet)#获取数据行与列数
    X = mat(ones((m,n)));   Y = mat(ones((m,1)))
    # 数据集矩阵的第一列初始化为1，偏置项；每个样本目标变量值存入Y
    X[:,1:n]=dataSet[:,0:n-1]; Y=dataSet[:,-1]
    xTx = X.T * X   #对数据集矩阵求内积
    if linalg.det(xTx) == 0.0: #计算行列式值是否为0，即判断是否可逆
        print('This matrix is singular, cannot do inverse,\n\
              try increasing the second value if ops')  #不可逆，打印信息
    ws = (xTx).I * (X.T*Y)  #可逆，计算回归系数
    return ws,X,Y   #返回回顾系数;数据集矩阵;目标变量值矩阵

#模型树的叶节点模型
def modelLeaf(dataSet):
    ws,X,Y = linearSolve(dataSet)   #调用线性回归函数生成叶节点模型
    return ws   #返回该叶节点线性方程的回归系数

#模型树的误差计算函数
def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)   #构建模型树叶节点的线性方程，返回参数
    yHat = X*ws #利用线性方程对数据集进行预测
    return sum(pow(y-yHat,2))   #返回误差的平方和，平方损失


#用树回归进行预测代码

#回归树的叶节点为float型常量
def regTreeEval(model, inDat):
    return float(model)

#模型树的叶节点浮点型参数的线性方程
def modelTreeEval(model, inDat):
    n = shape(inDat)[1]     #获取输入数据的列数
    X = mat(ones((1,n+1)))  #构建n+1维的单列矩阵
    X[:,1:n+1] = inDat  #第一列设置为1，线性方程偏置项b
    return float(X*model)  #返回浮点型的回归系数向量

#树预测
#@tree；树回归模型
#@inData：输入数据
#@modelEval：叶节点生成类型，需指定，默认回归树类型
def treeForeCast(tree,inData,modelEval=regTreeEval):
    # 如果当前树为叶节点，生成叶节点
    if not isTree(tree):    return modelEval(tree,inData)
    # 非叶节点，对该子树对应的切分点对输入数据进行切分
    if inData[tree['spInd']]>tree['spval']:
        if isTree(tree['left']):    #该树的左分支为非叶节点类型
            # 递归调用treeForeCast函数继续树预测过程，直至找到叶节点
            return treeForeCast(tree['left'],inData,modelEval)
        # 左分支为叶节点，生成叶节点
        else: return modelEval(tree['left'],inData)
    else:   # 小于切分点值的右分支
        if isTree(tree['right']):   #非叶节点类型
            # 继续递归treeForeCast函数寻找叶节点
            return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData) #叶节点，生成叶节点类型

#创建预测树
def createForeCast(tree, testData, modelEval=regTreeEval):
    m = len(testData)   #测试集样本数
    yHat = mat(zeros((m,1)))    #初始化行向量各维度值为1
    for i in range(m):  #遍历每个样本
        # 利用树预测函数对测试集进行树构建过程，并计算模型预测值
        yHat[i,0] = treeForeCast(tree,mat(testData[i]),modelEval)
    return yHat #返回预测值
