import numpy as np
from collections import defaultdict, deque

def checkGraphConnectivity(adjMatrix):

    numNodes = adjMatrix.shape[0]
    visited = np.zeros(numNodes, dtype=bool)
    queue = deque([0])
    visited[0] = True

    while queue:
        currentNode = queue.popleft()
        neighbors = np.where(adjMatrix[currentNode, :] > 0)[0]
        for neighbor in neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)

    isConnected = np.all(visited)
    
    return isConnected, visited

def groupmake_strict(A, groupsize=100):
    group_num = A.shape[0]
    group_vars = [set() for _ in range(group_num)]
    next_var_id = 0

    # 1. 强制共享变量
    for i in range(group_num):
        for j in range(i + 1, group_num):
            k = int(A[i, j])
            if k > 0:
                shared_vars = set(range(next_var_id, next_var_id + k))
                next_var_id += k
                group_vars[i].update(shared_vars)
                group_vars[j].update(shared_vars)

    # 2. 补齐私有变量
    for i in range(group_num):
        needed = groupsize - len(group_vars[i])
        if needed < 0:
            raise ValueError(
                f"group {i} has more than {groupsize} shared variables"
            )
        private_vars = set(range(next_var_id, next_var_id + needed))
        next_var_id += needed
        group_vars[i].update(private_vars)

    # 3. 重新编号到 [0, D)
    all_vars = sorted(set().union(*group_vars))
    var_map = {v: idx for idx, v in enumerate(all_vars)}

    allgroups = []
    for i in range(group_num):
        allgroups.append([var_map[v] for v in group_vars[i]])

    return allgroups
def computeRotation(D):
    A = np.random.randn(D, D)
    Q, R = np.linalg.qr(A)
    return Q, R

def data_generator(problem_batch_size,G_num,G_prob):

    topo_list = []
    w_list = []
    xopt_list = []
    xopt_1_list = []
    allgroups_list = []
    D_list = []
    R100_list = []
    group_num = G_num  # 10
    G_prob = G_prob    # 0.1
    
    for func in range(1, problem_batch_size + 1):

        np.random.seed(None)

        A = np.zeros((group_num, group_num))

        for i in range(group_num):
            for j in range(group_num):
                if i != j:
                    A[i, j] = np.random.rand() < G_prob  #链接概率

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

        allgroups = groupmake_strict(A, groupsize=100)
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
        "xopt_1_list": xopt_1_list,
        "D_list": D_list,
        "R100_list": R100_list,
    }

    # 使用结构化的字典格式返回，便于后续使用
    output = {
        'topology': np.array(topo_list),   # 拓扑矩阵
        'weights': np.array(w_list),       # 权重
        'details': detail                  # 详细信息
    }

    return output
