class UnionFind:


    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX


def merge_groups(allgroups, topo):
    n = len(allgroups)
    uf = UnionFind(n)
    merged_group_indices = {i: [i] for i in range(n)}

    for i in range(topo.shape[0]):
        for j in range(topo.shape[1]):
            if topo[i, j] == 1:
                root_i = uf.find(i)
                root_j = uf.find(j)
                if root_i != root_j:
                    uf.union(i, j)
                    new_root = uf.find(root_i)
                    merged_group_indices[new_root] = merged_group_indices.pop(
                        root_i, []
                    ) + merged_group_indices.pop(root_j, [])

    merged_groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in merged_groups:
            merged_groups[root] = set()
        merged_groups[root].update(allgroups[i])

    merged_groups = [list(group) for group in merged_groups.values()]
    merged_group_indices = list(merged_group_indices.values())

    return merged_groups, merged_group_indices