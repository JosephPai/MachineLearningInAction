from numpy import *

def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fr:
        for line in fr.readlines():
            curLine = line.strip().split()
            fltLine = list(map(float,curLine))#将每一行的数据映射成float型
            dataMat.append(fltLine)
    return dataMat

# 欧氏距离
def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))

#随机初始化K个质心(质心满足数据边界之内)
def randCent(dataSet, k):
    n = shape(dataSet)[1]   #得到数据样本的维度
    centroids = mat(zeros((k,n)))   #初始化为一个(k,n)的矩阵
    for j in range(n):  #遍历数据集的每一维度
        minJ = min(dataSet[:,j])    #得到该列数据的最小值
        rangeJ = float(max(dataSet[:,j]) - minJ)    #得到该列数据的范围(最大值-最小值)
        # k个质心向量的第j维数据值随机为位于(最小值，最大值)内的某一值
        centroids[:,j] = minJ + rangeJ*random.rand(k,1)
    return centroids    #返回初始化得到的k个质心向量

#k-均值聚类算法
#@dataSet:聚类数据集
#@k:用户指定的k个类
#@distMeas:距离计算方法，默认欧氏距离distEclud()
#@createCent:获得k个质心的方法，默认随机获取randCent()
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]   #获取数据集样本数
    clusterAssment = mat(zeros((m,2)))  #初始化一个(m,2)的矩阵
    centroids = createCent(dataSet,k)   #创建初始的k个质心向量
    clusterChanged = True   #聚类结果是否发生变化的布尔类型
    while clusterChanged:
    # 只要聚类结果一直发生变化，就一直执行聚类算法，直至所有数据点聚类结果不变化
        clusterChanged=False    #聚类结果变化布尔类型置为false
        for i in range(m):  #遍历数据集每一个样本向量
            minDist = inf;  minIndex=-1 #初始化最小距离最正无穷；最小距离对应索引为-1
            for j in range(k):  #循环k个类的质心
                # 计算数据点到质心的欧氏距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI<minDist:  #如果距离小于当前最小距离
                    # 当前距离定为当前最小距离；最小距离对应索引对应为j(第j个类)
                    minDist = distJI;   minIndex=j
            # 当前聚类结果中第i个样本的聚类结果发生变化：布尔类型置为true，继续聚类算法
            if clusterAssment[i,0] != minIndex: clusterChanged=True
            # 更新当前变化样本的聚类结果和平方误差
            clusterAssment[i,:] = minIndex,minDist**2
        # 打印k-均值聚类的质心
        print(centroids)
        # 遍历每一个质心
        for cent in range(k):
            # 将数据集中所有属于当前质心类的样本通过条件过滤筛选出来
            ptsInClust = dataSet[nonzero(clusterAssment[:,0].A==cent)[0]]
            # 计算这些数据的均值（axis=0：求列的均值），作为该类质心向量
            centroids[cent,:] = mean(ptsInClust, axis=0)
    # 返回k个聚类，聚类结果及误差
    return centroids, clusterAssment


#二分K-均值聚类算法
#@dataSet:待聚类数据集
#@k：用户指定的聚类个数
#@distMeas:用户指定的距离计算方法，默认为欧式距离计算
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]   #获得数据集的样本数
    clusterAssment = mat(zeros((m,2)))  #初始化一个元素均值0的(m,2)矩阵
    # 获取数据集每一列数据的均值，组成一个长为列数的列表
    centroid0 = mean(dataSet, axis=0).tolist()[0]
    centList =[centroid0] #当前聚类列表为将数据集聚为一类
    for j in range(m): #遍历每个数据集样本
        # 计算当前聚为一类时各个数据点距离质心的平方距离
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    # 循环，直至二分k-均值达到k类为止
    while (len(centList) < k):
        lowestSSE = inf     #将当前最小平方误差置为正无穷
        for i in range(len(centList)):  #遍历当前每个聚类
            # 通过数组过滤筛选出属于第i类的数据集合
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 对该类利用二分k-均值算法进行划分，返回划分后结果，及误差
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 计算该类划分后两个类的误差平方和
            sseSplit = sum(splitClustAss[:,1])
            # 计算数据集中不属于该类的数据的误差平方和
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            # 打印这两项误差值
            print("sseSplit, and notSplit: ",sseSplit,sseNotSplit)
            # 划分第i类后总误差小于当前最小总误差
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i #第i类作为本次划分类
                bestNewCents = centroidMat  #第i类划分后得到的两个质心向量
                bestClustAss = splitClustAss.copy() #复制第i类中数据点的聚类结果即误差值
                lowestSSE = sseSplit + sseNotSplit #将划分第i类后的总误差作为当前最小误差
        # 数组过滤筛选出本次2-均值聚类划分后类编号为1数据点，将这些数据点类编号变为
        # 当前类个数+1，作为新的一个聚类
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        # 同理，将划分数据集中类编号为0的数据点的类编号仍置为被划分的类编号，使类编号
        # 连续不出现空缺
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        # 打印本次执行2-均值聚类算法的类
        print('the bestCentToSplit is: ',bestCentToSplit)
        # 打印被划分的类的数据个数
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新质心列表中的变化后的质心向量
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids
        # 添加新的类的质心向量
        centList.append(bestNewCents[1,:].tolist()[0])
        # 更新clusterAssment列表中参与2-均值聚类数据点变化后的分类编号，及数据该类的误差平方
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE
    return mat(centList), clusterAssment


# 实战！
import urllib
import urllib.request
import json
def geoGrab(strAddress, city):
    apiStem = 'http://where.yahooapis.com/geocode?'
    params = {}
    params['flags'] = 'J'
    params['appid'] = 'ppp68N8t'
    params['location'] = '%s %s' % (strAddress,city)
    url_params = urllib.parse.urlencode(params)
    yahoolApi = apiStem + url_params
    print(yahoolApi)
    c = urllib.request.urlopen(yahoolApi)
    return json.loads(c.read())

from time import sleep
def massPlaceFind(fileName):
    fw = open('places.txt','w')
    with open(fileName) as fr:
        for line in fr.readlines():
            line = line.strip()
            lineArr = line.split()
            retDict = geoGrab(lineArr[1],lineArr[2])
            if retDict['ResultSet']['Error']==0:
                lat = float(retDict['ResultSet']['Results'][0]['latitude'])
                lng = float(retDict['ResultSet']['Results'][0]['longitude'])
                print("%s\t%f\t%f\n" % (line,lat,lng))
            else: print("error fetching")
            sleep(1)
    fw.close()

def distSLC(vecA, vecB):
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180)* \
        cos(pi*vecB[0,0]-vecA[0,0])/180
    return arccos(a+b)*6371.0

import matplotlib
import matplotlib.pyplot as plt
def clusterClubs(numClust=5):
    datList = []
    for line in open('palce.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s','o','^','8','p','d','v','h','>','<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect,label='ax0',**axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect,label='ax1',frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0],\
                    ptsInCurrCluster[:, 1].flatten().A[0], \
                                           marker = markerStyle , s = 90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], \
                myCentroids[:, 1].flatten().A[0], marker = 1 + 1, s = 300)
    plt.show()