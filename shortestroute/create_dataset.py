import numpy as np
import scipy as sp
from env.shortestpath import ShortestPathEnv
import networkx as nx

def create_dataset(num_steps, node_num=12):

    A = sp.sparse.random(node_num, node_num, density=0.5, format='csr')
    A.data[:] = 1
    A = A.todense()
    A = np.ma.array(A, mask=np.eye(node_num)).filled(fill_value=0).astype(int)
    # print("sparsity = %.2f" % (1 - np.sum(A)/A.size))
    env = ShortestPathEnv(nx.from_numpy_array(A, create_using=nx.DiGraph), 0, 5)

    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    # simulate to create trajectories
    while len(obss) < num_steps:
        for _ in range(10):
            ac = np.random.choice(np.arange(0, env.adj_mat.shape[0]))
            state, ret, terminal = env.step(ac)
            obss += [state]
            actions += [ac]
            stepwise_returns += [ret]
            if terminal:
                done_idxs += [len(obss)]
                returns += [0]
                env.reset()
                break

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # create the return-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # create the timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps, env