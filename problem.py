import numpy as np

class Problem:
    def __init__(self, topo, w, group, allgroups, xopt, D, R100):
        self.topo = topo
        self.w = w
        self.R100 = R100
        self.allgroups = allgroups
        self.xopt = xopt
        self.D = D
        self.group = group
        self.group_num = 10
        self.groupsize = 100

    def objective(self, pop):       
        fitness_values = np.zeros(pop.shape[0])  
        for i in range(pop.shape[0]):  
            x = pop[i]
            a1 = self.shift(x, self.xopt)
            z = a1
            fit = 0
            for j in range(self.group_num):
                current_group = np.array(self.allgroups[j])
                indices = current_group
                fit += self.w[j] * self.elliptic(np.dot(self.R100, z[indices]))
            fitness_values[i] = fit

        return fitness_values
    
    def elliptic(self, x):
        f = 1e6 ** np.linspace(0, 1, x.size)
        return np.dot(f, x ** 2)

    def sphere(self, x):
        return np.dot(x, x)

    def shift(self, x, xopt):
        return x - xopt


def P(topo, w, group, allgroups, xopt, D, R100):
    return Problem(topo, w, group, allgroups, xopt, D, R100)
