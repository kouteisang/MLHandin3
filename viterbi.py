import numpy as np
import math
from utils import *

def log(x):
    if x == 0:
        return float('-inf')
    return math.log(x)

def make_table(m, n):
    """Make a table with `m` rows and `n` columns filled with zeros."""
    return [[0] * n for _ in range(m)]

def compute_w_log(model, x):
    K = len(model.init_probs)
    N = len(x)
    w = make_table(K, N)
    for i in range(K):
        w[i][0] = log(model.init_probs[i]) + log(model.emission_probs[i][x[0]])
    for n in range(1, N):
        for k in range(K):
            compute_k_n = float('-inf')
            for j in range(K):
                compute_k_n = max(log(model.trans_probs[j][k]) + w[j][n-1], compute_k_n)
            w[k][n] = compute_k_n + log(model.emission_probs[k][x[n]])
    print("done")
    return w

def opt_path_prob_log(w):
    w = np.array(w)
    return max(w[:,w.shape[1]-1])

def backtrack_log(model, x, w):
    w = np.array(w)
    y = w.shape[1]
    z_N = np.argmax(w[:,y-1])
    path = []
    path.append(z_N)
    for n in range(y-2, -1, -1):
        list_k = []
        pathLen = len(path)
        for k in range(len(w)):
            # 这行代码的本质是看从哪一步到达的下一步的最优值
            list_k.append(log(model.emission_probs[path[pathLen-1]][x[n+1]]) + log(model.trans_probs[k][path[pathLen-1]]) + w[k][n])
        path.append(np.argmax(np.array(list_k)))
    path.reverse()
    return translate_indices_to_ann(path)
