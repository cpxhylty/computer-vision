class  KD_node:
    def __init__(self, elt=None, split=None, LL=None, RR=None):
        '''
        elt:数据点
        split:划分域
        LL, RR:节点的左儿子跟右儿子
        '''
        self.elt = elt
        self.split = split
        self.left = LL
        self.right = RR
    def createKDTree(root, data_list):
        '''
        root:当前树的根节点
        data_list:数据点的集合(无序)
        return:构造的KDTree的树根
        '''
        LEN = len(data_list)
        if LEN == 0:
            return
        # 数据点的维度
        dimension = len(data_list[0])
        # 方差
        max_var = 0
        # 最后选择的划分域
        split = 0
        for i in range(dimension):
            items = []
            for t in data_list:
                items.append(t[i])
            var = computeVariance(items)
            if var > max_var:
                max_var = var
                split = i
        #根据划分域的数据对数据点进行排序
        data_list.sort(key=lambda x: x[split])
        #选择下标为len / 2的点作为分割点
        elt = data_list[LEN/2]
        root = KD_node(elt,split)
        root.left = createKDTree(root.left, data_list[0:LEN/2])
        root.right = createKDTree(root.right, data_list[(LEN/2+1):LEN])
        return root

def computeVariance(arrayList):
    '''
    arrayList:存放的数据点
    return:返回数据点的方差
    '''
    for ele in arrayList:
        ele = float(ele)
    LEN = float(len(arrayList))
    array = numpy.array(arrayList)
    sum1 =  array.sum()
    array2 = array * array
    sum2 = array2.sum()
    mean = sum1 / LEN
    #    D[X] = E[x^2] - (E[x])^2
    variance = sum2 / LEN - mean**2
    return variance