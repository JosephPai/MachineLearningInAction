def loadExData():
    return [[1,1,1,0,0],
            [2,2,2,0,0],
            [1,1,1,0,0],
            [5,5,5,0,0],
            [1,1,0,2,2],
            [0,0,0,3,3],
            [0,0,0,1,1]]
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

from numpy import *
from numpy import linalg as la

#欧式距离相似度计算
def ecludSim(inA, inB):
    return 1.0/(1.0 + la.norm(inA - inB))

# 皮尔逊相关系数相似度计算
def pearsSim(inA, inB):
    if len(inA) < 3:    return 1.0
    return 0.5+0.5*corrcoef(inA, inB, rowvar=0)[0][1]

#余弦距离相似度计算
def cosSim(inA, inB):
    num = float(inA.T*inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5+0.5*(num/denom)

#未评级物品的评分预测函数
#@dataMat：数据矩阵
#@user：目标用户编号(矩阵索引，从0开始)
#@simMeans：相似度计算方法 默认余弦距离
#@item：未评分物品编号(索引，从0开始)
def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]   #获取数据矩阵的列数，即物品个数
    simTotal = 0.0; ratSimTotal = 0.0   #需要更新的两个相似度计算相关的值
    for j in range(n):  #遍历矩阵的每一列（遍历目标用户评价的物品列）
        userRating = dataMat[user,j]
        if userRating == 0: continue    #如果目标用户对该物品未评分，跳出本次循环
        # 用'logical_and'函数，统计目标列与当前列中在当前行均有评分的数据
        overLap = nonzero(logical_and(dataMat[:,item].A>0,\
                                      dataMat[:,j].A>0))[0]
        if len(overLap)==0: similarity=0    #如果不存在，则当前列于目标列相似性为0返回
        # 否则，计算这两列中均有评分的行之间的相似度
        else:   similarity = simMeas(dataMat[overLap,item],
                                     dataMat[overLap,j])
        print('the %d and %d similarity is: %f' % (item,j,similarity))
        simTotal += similarity  #更新两个变量的值
        ratSimTotal += similarity * userRating
    if simTotal == 0:   return 0
    else:   return ratSimTotal/simTotal


#推荐系统主体函数
#@dataMat：数据矩阵
#@user：目标用户编号(矩阵索引，从0开始)
#@N=3：保留的相似度最高的前N个菜肴，默认为3个
#@simMeas=cosSim：相似度计算方法 默认余弦距离
#@estMethod=standEst：评分方法，默认是standEst函数
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 从数据矩阵中找出目标用户user所用的未评分的菜肴的列
    unratedItems = nonzero(dataMat[user,:].A==0)[1]
    # 如果没有，表明所有菜肴均有评分
    if len(unratedItems)==0:    return 'you rated everything'
    itemScores=[]
    for item in unratedItems:  #遍历每一个未评分的矩阵列（未评分的菜肴）
        # 预估的评分采用默认的评分方法
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 将该菜肴及其预估评分加入数组列表
        itemScores.append((item, estimatedScore))
    # 利用sorted函数对列表中的预估评分由高到低排序，返回前N个菜肴
    return sorted(itemScores,\
                  key=lambda jj:jj[1], reverse=True)[:N]

#引入SVD的推荐引擎
#@dataMat：数据矩阵
#@user：目标用户索引
#@simMeans：相似度计算方法
#@item：目标菜肴索引
def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal=0.0
    U,Sigma,VT = la.svd(dataMat)    #svd进行奇异值分解
    Sig4 = mat(eye(4)*Sigma[:4])    #保留前四个奇异值，并将奇异值转化为方阵
    xformedItems = dataMat.T * U[:,:4]*Sig4.I   #将数据矩阵进行映射到低维空间
    for j in range(n):  #遍历矩阵的每一列
        userRating = dataMat[user,j]
        if userRating==0 or j==item: continue   #该用户当前菜肴未评分，则跳出本次循环
        # 否则，按照相似度计算方法进行评分
        similarity = simMeas(xformedItems[item,:].T,\
                             xformedItems[j,:].T)
        print('the %d and %d similarity is: %f' % (item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal == 0:   return 0
    else:   return ratSimTotal/simTotal



# 基于SVD的图像压缩
def printMat(inMat, thresh=0.8):
    for i in range(32):
        for k in range(32):
            if float(inMat[i,k]) > thresh:
                print(1,end='')
            else:   print(0,end='')     # 不换行的写法
        print(' ')

def imgCompress(numSV=3, thresh=0.8):
    myl = []
    for line in open('0_5.txt').readlines():
        newRow=[]
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    print('****original matrix*****')
    printMat(myMat,thresh)

    U,Sigma,VT = la.svd(myMat)
    SigRecon = mat(zeros((numSV, numSV)))
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
    reconMat = U[:,:numSV] * SigRecon * VT[:numSV,:]
    print("****reconstructed matrix using %d singular values*****" % numSV)
    printMat(reconMat, thresh)