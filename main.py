import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import Model
from tqdm.auto import trange
import scipy.io as sio
from scipy.io import loadmat
import numpy as np
from collections import defaultdict, deque
import multiprocessing
import time
import matplotlib.pyplot as plt
import networkx as nx
import os
from PIL import Image
import concurrent.futures
from itertools import chain
from data_utils import data_generator
#from graph_utils import plot_and_save_graphs, save_adjacency_matrices_as_graphs
from problem import P
from utils import merge_groups
from decc import DECC


def l1_normalization(ws):

    l1_norms = np.sum(np.abs(ws), axis=1, keepdims=True)
    normalized_ws = ws / l1_norms

    return normalized_ws


def repeat_with_interval(data, repeat_count, interval):
    return [data[i % len(data)] for i in range(repeat_count * interval)]


def run_decc_parallel(problem, group):
    print(f"Worker started with PID: {os.getpid()} for problem {problem}")   
    decc_optimizer = DECC(problem, group)
    return decc_optimizer.run()

def Env(topos, ws, groups, detail):
    """环境接口，用于并行运行优化任务
    Args:
        topos (torch.Tensor): 表示问题子组之间的重叠
        ws (torch.Tensor): 表示子问题的占比权重
        groups (torch.Tensor): 表示分组方式
    """

    topos, ws, groups = topos.to("cpu"), ws.to("cpu"), groups.to("cpu")
    
    repeat_count = 7  
    interval = 8      

    problems = [
    P(topo, w, group, allgroups, xopt, D, R100)
    for topo, w, group, allgroups, xopt, D, R100 in zip(
        topos,
        ws,
        groups,
        repeat_with_interval(detail["allgroups_list"], repeat_count, interval),
        repeat_with_interval(detail["xopt_list"], repeat_count, interval),
        repeat_with_interval(detail["D_list"], repeat_count, interval),
        repeat_with_interval(detail["R100_list"], repeat_count, interval),
    )
]
    
    all_futures = []


    with concurrent.futures.ProcessPoolExecutor(max_workers= 60) as executor:
        for problem_idx, problem in enumerate(problems):
            future = executor.submit(run_decc_parallel, problem, groups[problem_idx])
            all_futures.append(future)
        opts = [future.result() for future in concurrent.futures.as_completed(all_futures)]


    opts = torch.tensor(opts, device="cpu")  

    reward = -opts

    print(opts.mean().item())

    return reward.to(device)


def PPOloss(data, agent, clip=0.1, beta=3):

    topos, ws, action, old_prob, reward = data

    P = agent(topos, ws)

    entropy = -torch.sum(P * torch.log(P + 1e-10), dim=-1).mean() 

    new_prob = (action * P).sum(-1)

    ratio = new_prob / old_prob

    surr1 = ratio * reward
    surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * reward

    loss = -torch.min(surr1, surr2).mean() - beta * entropy

    return loss


@torch.no_grad()
def Buffer(agent, topos, ws, detail):

    topos, ws = torch.tensor(topos).to(device), torch.tensor(ws).to(device)
    
    n = topos.shape[-1]

    P = agent(topos, ws)

    P = P[None, :, :, :, :].repeat(1, reapeat, 1, 1, 1).reshape(-1, n, n, 2)

    topos = topos[None, :, :, :].repeat(1, reapeat, 1, 1).reshape(-1, n, n)
    
    ws = ws[None, :, :].repeat(1, reapeat, 1).reshape(-1, n)
    
    action = torch.multinomial(P.reshape(-1, 2), num_samples=1).view(-1, n, n)
    
    action = nn.functional.one_hot(action, num_classes=2)
    
    prob = (action * P).sum(-1)
    
    groups = action[:, :, :, 0]
    
    reward = Env(topos, ws, groups, detail)              
             
    reward = reward.reshape(-1, problem_batch_size)  
    
    reward_mean = reward.mean(dim=0, keepdim=True)  
    
    reward_min = reward.min(dim=0, keepdim=True).values  
    
    reward_max = reward.max(dim=0, keepdim=True).values  
    
    range_values = reward_max - reward_min
    
    range_values[range_values == 0] = 1e-6  

    reward1 = reward - reward_mean
    
    normalized_data = (reward - reward_min) / range_values
    
    reward = (reward - reward_mean).reshape(-1, 1, 1)  
    
    reward = normalized_data.reshape(-1, 1, 1)
    
    return TensorDataset(topos, ws, action, prob, reward)



if __name__ == "__main__":

    start_time = time.time()  

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    n = 10
    problem_batch_size = 8
    reapeat = 7
    train_batch_size = 14
    lr = 1e-4
    n_epoch = 200  # 100
    clip = 0.1
    reuse_time = 10

    agent = Model(10, 2).to(device)
    opt = optim.Adam(agent.parameters(), lr=lr)

    for k in trange(n_epoch):

        print(f"Epoch: {k}")  
        
        agent = agent.train()

        topos, ws, detail, topo_list, w_list = data_generator(problem_batch_size)

        normalized_ws = l1_normalization(ws)

        ws = normalized_ws
        dataset = Buffer(agent, topos, ws, detail)
        
        dl = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

        for g in range(reuse_time):
            print(f"Iteration: {g}")  
            for data in dl:
                loss = PPOloss(data, agent, clip=clip)

                opt.zero_grad()
                loss.backward()
                opt.step()

        if k % 5  == 0:  
            model_path = f"model_epoch_{k}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"Model saved at {model_path}")

        if k % 5  == 0 :  
            current_time = time.time()
            elapsed_time = current_time - start_time
            print(f"Epoch {k} completed. Elapsed time: {elapsed_time:.2f} seconds.")


    end_time = time.time()  
    execution_time = end_time - start_time  

    print(f"运行时间: {execution_time:.2f} 秒")

    torch.save(agent.state_dict(), "model_final.pth")
    print("Final model saved as model_final.pth")

