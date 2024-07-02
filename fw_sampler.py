import os
import time
import math
import torch as th
from torch.utils.data import DataLoader
import random
import numpy as np
import pickle as pk
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces
from tqdm import tqdm
import scipy


from ogb.nodeproppred import DglNodePropPredDataset
from utils.data_util import load_class_split

def generate_task(class_dict, N, K, Q):
    classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()
    pos_node_idx = []
    target_idx = []
    for i in classes:
        
        sampled_idx = np.random.choice(class_dict[i], K+Q, replace=False).tolist()
        pos_node_idx.extend(sampled_idx[:K])
        target_idx.extend(sampled_idx[K:])
    return np.array(pos_node_idx), np.array(target_idx), np.array(classes)

def generate_task_with_valid(class_dict, N, K, Q):
    classes = np.random.choice(list(class_dict.keys()), N, replace=False).tolist()
    pos_node_idx = []
    target_idx = []
    val_idx = []
    for i in classes:
        
        sampled_idx = np.random.choice(class_dict[i], K+K+Q, replace=False).tolist()
        pos_node_idx.extend(sampled_idx[:K])
        val_idx.extend(sampled_idx[K:K+K])
        target_idx.extend(sampled_idx[K+K:])
    return np.array(pos_node_idx), np.array(target_idx), np.array(val_idx), np.array(classes)

def get_fw_graphs(mode, class_split, dataset_name, N, K, Q, num_tasks, repeat):
    
    
    
    if mode == "train":
        assert repeat == 1
        
    graph_fn = f"./subgraphs/{dataset_name}_fw_{mode}_N_{N}_K_{K}_Q_{Q}_nt_{num_tasks}_rep_{repeat}.pkl"
    if os.path.exists(graph_fn):
        id_support_repeat, id_query_repeat, class_selected_repeat = pk.load(open(graph_fn,"rb"))
        
    else:
        os.makedirs('./subgraphs/', exist_ok=True)
    
        class_dict = class_split[mode]
        id_support_repeat = []
        id_query_repeat = []
        class_selected_repeat = []
        for rep in range(repeat):
            id_support = []
            id_query = []
            class_selected = []
            for bas in range(num_tasks):
                sup, que, cla = generate_task(class_dict, N, K, Q)
                id_support.append(sup)
                id_query.append(que)
                class_selected.append(cla)
            id_support_repeat.append(id_support)
            id_query_repeat.append(id_query)
            class_selected_repeat.append(class_selected)
        with open(graph_fn, "wb") as f:
            pk.dump((id_support_repeat, id_query_repeat, class_selected_repeat), f)
    return id_support_repeat, id_query_repeat, class_selected_repeat






if __name__ == "__main__":
    mode = "train"
    N = 10
    K = 10
    dataset_name = "ogbn-arxiv"
    
    Q = 10
    num_tasks = 2000
    repeat = 5
    
    dataset = DglNodePropPredDataset(dataset_name, root=f"./dataset/")
    graph, labels = dataset[0]
    labels = labels.view(-1)
    class_split = load_class_split(dataset_name, labels)
    
    repeat = 1 if mode=="train" else repeat
    get_fw_graphs(mode, class_split, dataset_name, N, K, Q, num_tasks, repeat)

def get_fw_graphs_with_valid(mode, class_split, dataset_name, N, K, Q, num_tasks, repeat):
    
    
    
    
    if mode == "train":
        assert repeat == 1
        
    graph_fn = f"./subgraphs/{dataset_name}_fw_{mode}_N_{N}_K_{K}_Q_{Q}_nt_{num_tasks}_rep_{repeat}_val.pkl"
    if os.path.exists(graph_fn):
        id_support_repeat, id_query_repeat, id_val_repeat, class_selected_repeat = pk.load(open(graph_fn,"rb"))
        
    else:
        os.makedirs('./subgraphs/', exist_ok=True)
    
        class_dict = class_split[mode]
        id_support_repeat = []
        id_query_repeat = []
        id_val_repeat = []
        class_selected_repeat = []
        for rep in range(repeat):
            id_support = []
            id_query = []
            id_val = []
            class_selected = []
            for bas in range(num_tasks):
                sup, que, val, cla = generate_task_with_valid(class_dict, N, K, Q)
                id_support.append(sup)
                id_query.append(que)
                id_val.append(que)
                class_selected.append(cla)
            id_support_repeat.append(id_support)
            id_query_repeat.append(id_query)
            id_val_repeat.append(id_val)
            class_selected_repeat.append(class_selected)
        with open(graph_fn, "wb") as f:
            pk.dump((id_support_repeat, id_query_repeat, id_val_repeat, class_selected_repeat), f)
    return id_support_repeat, id_query_repeat, id_val_repeat, class_selected_repeat
        
if __name__ == "__main__":
    mode = "test"
    N = 10
    K = 10
    dataset_name = "ogbn-arxiv"
    
    Q = 10
    num_tasks = 2000
    repeat = 5
    
    dataset = DglNodePropPredDataset(dataset_name, root=f"./dataset/")
    graph, labels = dataset[0]
    labels = labels.view(-1)
    class_split = load_class_split(dataset_name, labels)
    
    repeat = 1 if mode=="train" else repeat
    get_fw_graphs(mode, class_split, dataset_name, N, K, Q, num_tasks, repeat)
    