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

    def rastrigin(x):

        A = 10
        #x = T_irreg(x)
        #x = T_asy(x, beta=0.2)
        #x = T_diag(x, alpha=10)
        fit = A * (x.shape[0] - np.sum(np.cos(2 * np.pi * x), axis=0)) + np.sum(x ** 2, axis=0)
        return fit

    def sphere(self, x):
        return np.dot(x, x)
    
    def schwefel(x):

        #x = T_irreg(x)
       # x = T_asy(x, 0.2)

        cumulative_sum = np.cumsum(x, axis=0)       # shape: (D, N)
        fit = np.sum(cumulative_sum ** 2, axis=0)   # sum over D, keep N samples
        return fit
    

    def shift(self, x, xopt):
        return x - xopt
    
    def T_irreg(x):
        c1 = 0.05
        c2 = 0.5
        c3 = 1e-4
        y = np.zeros_like(x)
        idx = x > 0
        y[idx] = np.log(x[idx]) / c1
        y[~idx] = -np.log(-x[~idx]) / c1
        y = np.sign(x) * np.exp(y + 0.49 * (np.sin(c2 * y) + np.sin(c3 * y)))
        return y
    
    def T_asy(x, beta):
    # x: shape (D, N)
        D = x.shape[0]
        idx = x > 0
        exponent = 1 + beta * np.linspace(0, 1, D).reshape(D, 1) * np.sqrt(x[idx])
        x[idx] = x[idx] ** exponent
        return x

    def T_diag(x, alpha):
    # x: shape (D, N)
        D = x.shape[0]
        coeffs = alpha ** (0.5 * np.linspace(0, 1, D)).reshape(D, 1)
        return coeffs * x


def P(topo, w, group, allgroups, xopt, D, R100):
    return Problem(topo, w, group, allgroups, xopt, D, R100)