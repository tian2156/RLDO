import numpy as np
from collections import defaultdict, deque

def checkGraphConnectivity(adjMatrix):

    numNodes = adjMatrix.shape[0]
    visited = np.zeros(numNodes, dtype=bool)
    queue = deque([0])
    visited[0] = True

    while queue:
        currentNode = queue.popleft()
        neighbors = np.where(adjMatrix[currentNode, :] == 1)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    isConnected = np.all(visited)
    
    return isConnected, visited

def groupmake(A):

    group_num = A.shape[0]
    allgroups = []
    selected_v = []
    numbershare = np.zeros((group_num, group_num))
    for i in range(group_num):
        startNum = 100 * i
        endNum = 100 * (i + 1) - 1
        allgroups.append(list(range(startNum, endNum + 1)))

    for i in range(group_num):
        for j in range(i + 1, group_num):
            if A[i, j] != 0:
                randomNumber = A[i, j]
                numbershare[i, j] = randomNumber
                filteredA = [x for x in allgroups[i] if x not in selected_v]
                indices = np.random.choice(filteredA, randomNumber, replace=False)
                selected_v.extend(indices)
                allgroups[j].extend(indices)
    selected_v = list(set(selected_v))

    for i in range(len(allgroups)):
        currentArray = allgroups[i]
        filteredArray = [x for x in currentArray if x in selected_v]
        remainingElements = [x for x in currentArray if x not in selected_v]
        additionalCount = 100 - len(filteredArray)
        indices = np.random.choice(remainingElements, additionalCount, replace=False)
        filteredArray.extend(indices)
        allgroups[i] = filteredArray

    allElements = [item for sublist in allgroups for item in sublist]
    allElements = list(set(allElements))
    lengh_allElements = len(allElements)
    _, newIndex = np.unique(allElements, return_inverse=True)
    valueMap = dict(zip(allElements, newIndex))

    for i in range(len(allgroups)):
        currentArray = [valueMap[val] for val in allgroups[i]]
        allgroups[i] = currentArray

    return allgroups

def computeRotation(D):
    A = np.random.randn(D, D)
    Q, R = np.linalg.qr(A)
    return Q, R

def data_generator(problem_batch_size):

    topo_list = []
    w_list = []
    xopt_list = []
    xopt_1_list = []
    allgroups_list = []
    D_list = []
    R100_list = []
    group_num = 10
    
    for func in range(1, problem_batch_size + 1):

        np.random.seed(None)

        A = np.zeros((group_num, group_num))

        for i in range(group_num):
            for j in range(group_num):
                if i != j:
                    A[i, j] = np.random.rand() < 0.1  #链接概率

        for i in range(group_num):
            if np.all(A[i, :] == 0) and np.all(A[:, i] == 0):
                numbers = list(range(group_num))
                numbers.remove(i)
                selected = np.random.choice(numbers)
                A[i, selected] = 1

        for i in range(group_num):
            for j in range(group_num):
                if i != j and A[i, j] == 1:
                    A[j, i] = 1

        connect, visited = checkGraphConnectivity(A)

        while not connect:           
            # 随机选择一个未访问的节点
            unvisitedIndices = np.where(~visited)[0]
            selected_node = np.random.choice(unvisitedIndices)          
            # 随机选择一个已访问的节点
            visitedIndices = np.where(visited)[0]  # 获取所有已访问的节点
            selected_neighbor = np.random.choice(visitedIndices)  # 从已访问的节点中随机选择一个节点
            # 将未访问节点与已访问节点连接
            A[selected_node, selected_neighbor] = 1
            A[selected_neighbor, selected_node] = 1  # 无向图，双向连接
            # 重新检查图的连通性
            connect, visited = checkGraphConnectivity(A)

        A = A.astype(int)
        A = np.triu(A)
        
        for i in range(group_num):
            for j in range(group_num):
                if A[i, j] == 1:
                    A[i, j] = np.random.randint(1, 11)
                    
        topo_list.append(A)

        w = np.zeros(group_num)
        index = np.random.permutation(group_num)
        for i in range(group_num):
            w[i] = 10 ** (3 * np.random.normal(0, 1))  #CEC 2010
        w_list.append(w)

        allgroups = groupmake(A)
        allgroups_list.append(allgroups)
        allElements = list(set([item for sublist in allgroups for item in sublist]))
        D_list.append(len(allElements))
        
        lengh_allElements = len(allElements)
        xopt = -100 + 2 * 100 * np.random.rand(lengh_allElements)
        xopt_1 = -100 + 2 * 100 * np.random.rand(1000)
        xopt_list.append(xopt)
        xopt_1_list.append(xopt_1)

    
        R100, _ = computeRotation(100)
        R100_list.append(R100)

    detail = {
        "allgroups_list": allgroups_list,
        "xopt_list": xopt_list,
        "xopt_1_list":xopt_1_list,
        "D_list": D_list,
        "R100_list": R100_list,
    }

    return np.array(topo_list), np.array(w_list), detail, topo_list, w_list
