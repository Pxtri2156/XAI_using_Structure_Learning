import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
import igraph as ig
import random
import torch
import networkx as nx 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm 
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from collections import deque
import time

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm

def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W

def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
    return X

def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X

def read_data(data_path, out_folder, ratio_test):
    X = pd.read_csv(data_path)
    X = X.to_numpy()[:,1:].astype(float)
    np.random.shuffle(X)
    split_n = int((1-ratio_test)*X.shape[0])
    X_train = X[:split_n,:]
    X_test = X[split_n:, :]
    np.savetxt(out_folder+'X.csv', X, delimiter=',')
    np.savetxt(out_folder+'X_train.csv', X_train, delimiter=',')
    np.savetxt(out_folder+'X_test.csv', X_test, delimiter=',')
    return X, X_train, X_test

def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}

def draw_directed_graph(W_est, graph_vis_path, labels):
    
    G = nx.DiGraph() 
    rows, cols = np.where(W_est != 0)
    edges = zip(rows.tolist(), cols.tolist())
    G.add_nodes_from(range(W_est.shape[0]))
    G.add_edges_from(edges) 
    
    #normalize 
    W_est_normal = (W_est - np.min(W_est)) / (np.max(W_est) - np.min(W_est))

    pos = nx.circular_layout(G)

    #DESIGN 
    edge_widths = [W_est[u][v] for u, v in G.edges()]
    # edge_widths = [W_est_normal[u][v]*2 for u, v in G.edges()] ## normalization weight

    # edge label
    edge_labels = {(u, v): round(W_est[u][v],2) for u, v in G.edges()}
 
    node_colors = []
    for node in G.nodes(data=True):
        if node[0] == len(labels.keys())-1: 
            node_colors.append([1.0, 0.8, 0.7])
        else:
            node_colors.append([0.7, 0.85, 1.0])
    
    #DRAW
    nx.draw_networkx(G, pos, with_labels=True, labels=labels, node_shape='o', node_size=1200, node_color=node_colors, font_size=9) 
    nx.draw_networkx_edges(G, pos, width=edge_widths)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="black", font_size=7)
    plt.savefig(graph_vis_path)
    plt.clf()
    G.clear()
    
def get_labels(name):
    if name == 'ov_cancer': 
        labels = {0:'0:BRCA1',1:'1:CDK12',2:'2:KRAS',3:'3:NF1',4:'4:NRAS',5:'5:RB1',6:'6:TP53',7:'7:ZNF133',8:'8:OV'}
        
    elif name == 'laml_cancer':
        labels = {0:'0:ASXL1',1:'1:DNMT3A',2:'2:FLT3',3:'3:IDH1',4:'4:IDH2',5:'5:KIT',
                       6:'6:KRAS',7:'7:NPM1',8:'8:PTPDC1',9:'9:PTPN11',10:'10:RUNX1',11:'11:SF3B1',
                       12:'12:SMC1A',13:'13:TP53', 14:'14:U2AF1', 15:'15:WT1', 16:'16:LAML'}
    elif name == 'laml_cancer_non_filter':
        labels = {0:'0:ASXL1',1:'1:DNMT3A',2:'2:FLT3',3:'3:IDH1',4:'4:IDH2',5:'5:KIT',
                       6:'6:KRAS',7:'7:NPM1',8:'8:PTPDC1',9:'9:PTPN11',10:'10:RUNX1',11:'11:SF3B1',
                       12:'12:SMC1A',13:'13:TP53', 14:'14:U2AF1', 15:'15:WT1', 16:"16:LATS1",
                       17:"17:DACH1", 18:"18:TGIF1", 19:'19:LAML'}
    elif name == 'ov_cancer_non_filter': 
        labels = {0:'0:BRCA1',1:'1:CDK12',2:'2:KRAS',3:'3:NF1',4:'4:NRAS',5:'5:RB1',6:'6:TP53',7:'7:ZNF133',
                  8:"8:TGIF1", 9:"9:WHSC1", 10:"10:CREB3L3", 11:"11:POLRMT", 12:"12:SMARCB1", 13:"13:DIAPH2",
                  14:'14:OV'}
    
    elif name == 'stad_cancer':
        labels = {0:'0:APC', 1:'1:ARID1A', 2:'2:ARID2', 3:'3:BCOR', 4:'4:CASP8', 5:'5:CDH1',
            6:'6:CDKN2A', 7:'7:CTNNB1', 8:'8:DMD', 9:'9:ERBB2', 10:'10:FBXW7', 11:'11:KRAS',
            12:'12:MUC6', 13:'13:PIK3CA', 14:'14:PTEN', 15:'15:RASA1', 16:'16:RHOA', 17:'17:RNF43', 18:'18:SMAD2',
            19:'19:SMAD4', 20:'20:TP53', 21:'21:STAD'}
        
    elif name == "adult_income":
        labels = {0:'0:hours', 1:'1:edu', 2:'2:job', 3:'3:w_class',
                  4: '4:race', 5: '5:age', 6:'6:marry', 7:'7:sex', 8:'8:income'}
        
    elif name == "boston_housing":
        labels = {0:'0:CRIM', 1:'1:INDUS', 2:'2:NOX', 3:'3:RM', 4: '4:AGE', 5:'5:DIS', 6:'6:RAD', 
                  7:'7:TAX', 8:'8:PTRATIO', 9:'9:B', 10:'10:LSTAT', 11:'11:MEDV'}
    
    elif name == "cali_housing":
        labels = {0:'0:MedInc', 1:'1:HouseAge', 2:'2:AveRooms', 3:'3:AveBedrms', 
                  4:'4:Population', 5:'5:AveOccup', 6:'6:Latitude', 7:'7:Longitude', 8:'8:price'}
    
    elif name == "breast_cancer":
        labels = {0:'0:age', 1:'1:menopause', 2:'2:tumor-size', 3:'3:inv-nodes', 4:'4:node-caps',
                  5:'5:deg-malig', 6:'6:breast', 7:'7:breast-quad', 8:'8:irradiat', 9:'9:Cancer'}
        
    elif name == 'diabetes':
        labels= {0:'0:age', 1:'1:sex', 2:'2:bmi', 3:'3:bp', 4:'4:s1',
                 5:'5:s2', 6:'6:s3', 7:'7:s4', 8:'8:s5', 9:'9:s6', 10:'10:target'}
        
    elif name == 'friedman1':
        labels = {0:'0:ft_0', 1:'1:ft_1', 2:'2:ft_2', 3:'3:ft_3',
                 4:'4:ft_4', 5:'5:ft_5', 6:'6:ft_6', 7:'7:ft_7', 8:'8:ft_8', 9:'ft_9', 10:'10:target'}
        
    elif name == 'parkinson':
        label = {0:'0',1:'1', 2:'2', 3:'3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
                 10:'10', 11:'11', 12:'12', 13:'13', 14:'14', 15:'15',16:'16',17:'17',
                 18:'18', 19:'19', 20:'20', 21:'21', 22:'22', 23:'23', 24:'24', 25:'25',
                 26:'26', 27:'27', 28:'28', 29:'29', 30:'30', 32:'31', 32:'32', 33:'33', 34:'34',
                 35:'35', 36:'36', 37:'37', 38:'38', 39:'39', 40:'40', 41:'41', 42:'42', 43:'43',
                 44:'44', 45:'45', 46:'46', 47:'47', 48:'48', 49:'49', 50:'50', 51:'51',
                 52:'52', 53:'53', 54:'54', 55:'55', 56:'56', 57:'57', 58:'58', 59:'59', 60:'60'}
    else: 
        raise  Exception("Dataset name is not defined labels")
        
    return  labels

def draw_head_map(array, save_path): 
    ax = sns.heatmap(array, linewidth=0.5, cmap='rocket_r')
    plt.savefig(save_path)
    plt.clf()
    
def draw_seed_graphs(root_path, w_threshold, no_seeds, labels):    
    # Get avarage W_est from random seeds 
    for seed in tqdm(range(no_seeds)):
        seed_name = 'seed_'+str(seed)
        out_folder = root_path + f"{seed_name}"
        if not os.path.isdir(out_folder):
            os.mkdir(out_folder)
        out_folder += "/"
        
        # Load seed data 
        seed_result_path = out_folder + 'W_est.csv'
        print(seed_result_path)
        seed_W_est = np.loadtxt(seed_result_path, delimiter=",", dtype=float)
        seed_W_est[np.abs(seed_W_est) < w_threshold] = 0
        assert is_dag(seed_W_est)
        
        ## Visualize graph 
        vis_path = out_folder + f"vis_avg_graph_{str(w_threshold)}.png"
        draw_directed_graph(seed_W_est, vis_path, labels) 
        
def partial_dependence_plot(X, model, save_path, selected_features, dataset_name):
    features = range(0,X.shape[1])
    # Train the model
    # Create PartialDependenceDisplay object

    if dataset_name == 'breast_cancer':
        pdp_display = PartialDependenceDisplay.from_estimator(model, X, features=features, categorical_features=features, feature_names=selected_features[:-1], kind='average')
    else:
        pdp_display = PartialDependenceDisplay.from_estimator(model, X, features=features, feature_names=selected_features[:-1], kind='both')

    # Create the partial dependence plot
    fig, ax = plt.subplots(figsize=(10, 8))
    pdp_display.plot(ax=ax, line_kw={'alpha': 0.1}, 
                    pd_line_kw={'linewidth': 4, 'linestyle': '-', 'alpha': 0.8})
    plt.tight_layout()
    plt.legend().remove()
    # Save the plot as an image
    file_name = f'pdp_{dataset_name}.png'
    plt.savefig(os.path.join(save_path, file_name))    

def cal_pdp(model, X, feature_names):
    features = range(0,X.shape[1])
    pdp = partial_dependence(model, features=features, X=X, feature_names=feature_names, percentiles=(0, 1), grid_resolution=2) 
    return pdp 

def shortest_path(adj_matrix, start, end):
    if not 0 <= start < len(adj_matrix) or not 0 <= end < len(adj_matrix):
        raise ValueError("Invalid start or end node")

    # Initialize visited array to keep track of visited nodes
    visited = [False] * len(adj_matrix)

    # Initialize queue for BFS and enqueue the start node
    queue = deque([(start, [])])

    while queue:
        current_node, path = queue.popleft()
        visited[current_node] = True

        for neighbor, has_path in enumerate(adj_matrix[current_node]):
            if has_path == 1 and not visited[neighbor]:
                if neighbor == end:
                    # Return the shortest path
                    return path + [current_node, neighbor]

                # Enqueue the neighbor with the updated path
                queue.append((neighbor, path + [current_node]))

    # If no path is found
    return None

def get_contribution(adjacency_matrix):
    vertices = len(adjacency_matrix[0]) - 1 
    end_node = vertices
    contribution_dic = {}
    for start_node in range(vertices):
        result = shortest_path(adjacency_matrix, start_node, end_node)
        if result:
            contribution_values = 1/(len(result)-1)
        else:
            contribution_values=0 
        contribution_dic[str(start_node)] = contribution_values
    return contribution_dic

def matrix_factorization_svd(W, t):
    # Perform SVD on matrix W
    U, S, V = torch.svd(W)
    
    # Take the first t singular values and vectors
    Ut = U[:, :t]
    St = torch.diag(S[:t])
    Vt = V[:, :t]
    
    # Construct matrix A and B
    A = Ut @ torch.sqrt(St)
    B = torch.sqrt(St) @ Vt.t()
    return A, B

def series(x, eps = 1e-8):
    """
    compute the matrix series: \sum_{k=0}^{\infty}\frac{x^{k}}{(k+1)!}
    """
    s = torch.eye(x.size(-1), dtype=torch.double,  device=x.device)
    t = x / 2
    k = 3
    while torch.norm(t, p=1, dim=-1).max().item() > eps:
        s = s + t
        t = torch.matmul(x, t) / k
        k = k + 1
    return s

def cal_expm(A, B, I):
    """
    Compute the expm of W, W=A.B based on A, B
    """
    V = B.matmul(A)
    series_V = series(V)
    expm_W = I + (A.matmul(series_V)).matmul(B)
    # expm_W = expm_W.numpy()
    return expm_W

def benchmark_me():
    def decorator(function):
        def wraps(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            end = time.time()
            print(f"\t\t{function.__name__}: exec time: {(end - start) *10**3}")
            # wandb.log({function.__name__:(end - start) *10**3})
            return result
        return wraps
    return decorator