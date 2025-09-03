import numpy as np
from utils import merge_groups


class DECC:
    def __init__(self, problem, group, NP=50, Max_FEs=300000):

        np.random.seed(None)  # 使用系统时间或其他熵源生成随机种子
        self.problem = problem
        self.Max_FEs = Max_FEs
        self.group = group
        self.allgroups = problem.allgroups
        self.merged_groups, self.merged_group_indices = merge_groups(
            self.allgroups, self.group
        )
        self.NP = NP

    def run(self):

        FEs = self.Max_FEs
        D = self.problem.D  # 维度
        Lbound = np.full((self.NP, D), -100)  # 下界
        Ubound = np.full((self.NP, D), 100)  # 上界
        group = self.merged_groups

        bestval = self.decc(D, Lbound, Ubound, FEs, group)

        return bestval

    def decc(self, D, Lbound, Ubound, FEs, group):
        # 初始化种群
        pop = Lbound + np.random.rand(self.NP, D) * (Ubound - Lbound)
        # 评估初始种群的适应度
        val = self.problem.objective(pop)
        # val1 = self.problem.objective(self.problem.xopt.reshape(1, D))
        
        FEs = FEs - self.NP
        Max_Gen = FEs / self.NP
        bestval = np.min(val)
        bestmem = pop[np.argmin(val), :]
        # 差分分组
        #group = self.merged_groups
        group_num = len(group)

        iter = 0
        while iter < Max_Gen:
            for i in range(group_num):
                dim_index = group[i]  # 子分量索引
                subpop = pop[:, dim_index]
                subLbound = Lbound[:, dim_index]
                subUbound = Ubound[:, dim_index]

                # 子组件优化 (DE)
                subpopnew, bestmemnew, bestvalnew = self.de(
                    dim_index, subpop, bestmem, bestval, subLbound, subUbound
                )
                # 更新种群和最佳值
                pop[:, dim_index] = subpopnew
                bestmem = bestmemnew
                bestval = bestvalnew

                iter = iter + group_num
                if iter >= Max_Gen:
                    break

        return bestval



    def de(self, dim_index, subpop, bestmem, bestval, Lbound, Ubound):

        F = 0.5  # 变异因子
        CR = 0.3  # 交叉概率
        NP, D = subpop.shape  # 种群大小和维度

        subpopnew = np.copy(subpop)

        # 初始适应度计算
        gpop = np.ones((NP, 1)) * bestmem  # 全局种群初始化为最佳个体
        gpop[:, dim_index] = subpop
        val = self.problem.objective(gpop)
        bestvalnew = np.min(val)
        bestmemnew = gpop[np.argmin(val), :]

        # 差分进化的主循环
        for i in range(NP):
            idxs = [idx for idx in range(NP) if idx != i]
            a, b, c = subpop[np.random.choice(idxs, 3, replace=False)]
            donor = np.clip(a + F * (b - c), Lbound[i], Ubound[i])  # 变异操作

            cross_points = np.random.rand(D) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, D)] = True  # 保证至少一个维度被交叉

            trial = np.where(cross_points, donor, subpop[i])
            trial = np.clip(trial, Lbound[i], Ubound[i])  # 保证 trial 在边界内

            gpop_trial = np.copy(gpop)
            gpop_trial[:, dim_index] = subpop
            gpop_trial[i, dim_index] = trial  # 更新 i 个体

            trial_val = self.problem.objective(gpop_trial)[i]

            if trial_val < val[i]:
                subpopnew[i] = trial
                val[i] = trial_val

                if trial_val < bestvalnew:
                    bestvalnew = trial_val
                    bestmemnew = (gpop_trial)[i]

        return subpopnew, bestmemnew, bestvalnew
