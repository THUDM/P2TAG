import os
import time
import math
import torch as th
from torch.utils.data import DataLoader
import random
import numpy as np
import dgl.function as fn
import dgl
from dgl.sampling import random_walk, pack_traces
from tqdm import tqdm
import scipy



class SAINTSampler:
    """
    Description
    -----------
    SAINTSampler implements the sampler described in GraphSAINT. This sampler implements offline sampling in
    pre-sampling phase as well as fully offline sampling, fully online sampling in training phase.
    Users can conveniently set param 'online' of the sampler to choose different modes.
    Parameters
    ----------
    node_budget : int
        the expected number of nodes in each subgraph, which is specifically explained in the paper. Actually this
        param specifies the times of sampling nodes from the original graph with replacement. The meaning of edge_budget
        is similar to the node_budget.
    dn : str
        name of dataset.
    g : DGLGraph
        the full graph.
    train_nid : list
        ids of training nodes.
    num_workers_sampler : int
        number of processes to sample subgraphs in pre-sampling procedure using torch.dataloader.
    num_subg_sampler : int, optional
        the max number of subgraphs sampled in pre-sampling phase for computing normalization coefficients in the beginning.
        Actually this param is used as ``__len__`` of sampler in pre-sampling phase.
        Please make sure that num_subg_sampler is greater than batch_size_sampler so that we can sample enough subgraphs.
        Defaults: 10000
    batch_size_sampler : int, optional
        the number of subgraphs sampled by each process concurrently in pre-sampling phase.
        Defaults: 200
    online : bool, optional
        If `True`, we employ online sampling in training phase. Otherwise employing offline sampling.
        Defaults: True
    num_subg : int, optional
        the expected number of sampled subgraphs in pre-sampling phase.
        It is actually the 'N' in the original paper. Note that this param is different from the num_subg_sampler.
        This param is just used to control the number of pre-sampled subgraphs.
        Defaults: 50
    full : bool, optional
        True if the number of subgraphs used in the training phase equals to that of pre-sampled subgraphs, or
        ``math.ceil(self.train_g.num_nodes() / self.node_budget)``. This formula takes the result of A divided by B as
        the number of subgraphs used in the training phase, where A is the number of training nodes in the original
        graph, B is the expected number of nodes in each pre-sampled subgraph. Please refer to the paper to check the
        details.
        Defaults: True
    Notes
    -----
    For parallelism of pre-sampling, we utilize `torch.DataLoader` to concurrently speed up sampling.
    The `num_subg_sampler` is the return value of `__len__` in pre-sampling phase. Moreover, the param `batch_size_sampler`
    determines the batch_size of `torch.DataLoader` in internal pre-sampling part. But note that if we wanna pass the
    SAINTSampler to `torch.DataLoader` for concurrently sampling subgraphs in training phase, we need to specify
    `batch_size` of `DataLoader`, that is, `batch_size_sampler` is not related to how sampler works in training procedure.
    """

    def __init__(self, node_budget, dn, g, num_workers_sampler, num_subg_sampler=10000,
                 batch_size_sampler=200, online=True, num_subg=50, full=True):
        self.g = g.cpu()
        self.node_budget = node_budget
        
        self.train_g = g
        self.dn, self.num_subg = dn, num_subg
        self.node_counter = th.zeros((self.train_g.num_nodes(),))
        self.edge_counter = th.zeros((self.train_g.num_edges(),))
        self.prob = None
        self.num_subg_sampler = num_subg_sampler
        self.batch_size_sampler = batch_size_sampler
        self.num_workers_sampler = num_workers_sampler
        self.train = False
        self.online = online
        self.full = full

        assert self.num_subg_sampler >= self.batch_size_sampler, "num_subg_sampler should be greater than batch_size_sampler"
        graph_fn, norm_fn = self.__generate_fn__()

        if os.path.exists(graph_fn):
            self.subgraphs = np.load(graph_fn, allow_pickle=True)
            
        else:
            os.makedirs('./subgraphs/', exist_ok=True)

            self.subgraphs = []
            self.N, sampled_nodes = 0, 0
            

            
            loader = DataLoader(self, batch_size=self.batch_size_sampler, shuffle=True,
                                num_workers=self.num_workers_sampler, collate_fn=self.__collate_fn__, drop_last=False)

            t = time.perf_counter()
            for num_nodes, subgraphs_nids, subgraphs_eids in tqdm(loader, desc="preprocessing saint subgraphs"):

                self.subgraphs.extend(subgraphs_nids)
                sampled_nodes += num_nodes

                _subgraphs, _node_counts = np.unique(np.concatenate(subgraphs_nids), return_counts=True)
                sampled_nodes_idx = th.from_numpy(_subgraphs)
                _node_counts = th.from_numpy(_node_counts)
                self.node_counter[sampled_nodes_idx] += _node_counts

                _subgraphs_eids, _edge_counts = np.unique(np.concatenate(subgraphs_eids), return_counts=True)
                sampled_edges_idx = th.from_numpy(_subgraphs_eids)
                _edge_counts = th.from_numpy(_edge_counts)
                self.edge_counter[sampled_edges_idx] += _edge_counts

                self.N += len(subgraphs_nids)  

                
                if sampled_nodes > self.train_g.num_nodes() * num_subg:
                    break

            print(f'Sampling time: [{time.perf_counter() - t:.2f}s]')
            np.save(graph_fn, self.subgraphs)

            
            
            
            

        
        
        

        random.shuffle(self.subgraphs)
        self.__clear__()
        print("The number of subgraphs is: ", len(self.subgraphs))

        self.train = True

    def __len__(self):
        if self.train is False:
            return self.num_subg_sampler
        else:
            if self.full:
                return len(self.subgraphs)
            else:
                return math.ceil(self.train_g.num_nodes() / self.node_budget)

    def __getitem__(self, idx):
        
        
        if self.train:
            if self.online:
                subgraph = self.__sample__()
                return dgl.node_subgraph(self.train_g, subgraph)
            else:
                return dgl.node_subgraph(self.train_g, self.subgraphs[idx])
        else:
            subgraph_nids = self.__sample__()
            num_nodes = len(subgraph_nids)
            subgraph_eids = dgl.node_subgraph(self.train_g, subgraph_nids).edata[dgl.EID]
            return num_nodes, subgraph_nids, subgraph_eids

    def __collate_fn__(self, batch):
        if self.train:  
            return batch[0]
        else:
            sum_num_nodes = 0
            subgraphs_nids_list = []
            subgraphs_eids_list = []
            for num_nodes, subgraph_nids, subgraph_eids in batch:
                sum_num_nodes += num_nodes
                subgraphs_nids_list.append(subgraph_nids)
                subgraphs_eids_list.append(subgraph_eids)
            return sum_num_nodes, subgraphs_nids_list, subgraphs_eids_list

    def __clear__(self):
        self.prob = None
        self.node_counter = None
        self.edge_counter = None
        self.g = None

    def __generate_fn__(self):
        raise NotImplementedError

    def __compute_norm__(self):

        self.node_counter[self.node_counter == 0] = 1
        self.edge_counter[self.edge_counter == 0] = 1

        loss_norm = self.N / self.node_counter / self.train_g.num_nodes()

        self.train_g.ndata['n_c'] = self.node_counter
        self.train_g.edata['e_c'] = self.edge_counter
        self.train_g.apply_edges(fn.v_div_e('n_c', 'e_c', 'a_n'))
        aggr_norm = self.train_g.edata.pop('a_n')

        self.train_g.ndata.pop('n_c')
        self.train_g.edata.pop('e_c')

        return aggr_norm.numpy(), loss_norm.numpy()

    def __compute_degree_norm(self):

        self.train_g.ndata['train_D_norm'] = 1. / self.train_g.in_degrees().float().clamp(min=1).unsqueeze(1)
        self.g.ndata['full_D_norm'] = 1. / self.g.in_degrees().float().clamp(min=1).unsqueeze(1)

    def __sample__(self):
        raise NotImplementedError
    
    def get_generated_subgraph_nodes(self):
        return self.subgraphs



class SAINTRandomWalkLoader(SAINTSampler):
    """
    Description
    -----------
    GraphSAINT with random walk sampler
    Parameters
    ----------
    num_roots : int
        the number of roots to generate random walks.
    length : int
        the length of each random walk.
    """
    def __init__(self, feats, num_roots, length, **kwargs):
        self.num_roots, self.length = num_roots, length
        self.feats = feats
        super(SAINTRandomWalkLoader, self).__init__(node_budget=num_roots * length, **kwargs)

    def __generate_fn__(self):
        graph_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_{}.npy'.format(self.dn, self.num_roots,
                                                                        self.length, self.num_subg, self.num_subg_sampler))
        norm_fn = os.path.join('./subgraphs/{}_RW_{}_{}_{}_{}_norm.npy'.format(self.dn, self.num_roots,
                                                                            self.length, self.num_subg, self.num_subg_sampler))
        return graph_fn, norm_fn

    def __sample__(self):
        sampled_roots = th.randint(0, self.train_g.num_nodes(), (self.num_roots,))
        traces, types = random_walk(self.train_g, nodes=sampled_roots, length=self.length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        sampled_nodes = sampled_nodes.unique()
        return sampled_nodes.numpy()

    def __collate_fn__(self, batch):
        if self.train:  
            subg = batch[0]
            node_ids = subg.ndata["_ID"]
            feats = self.feats[node_ids]
            return feats, subg
        else:
            sum_num_nodes = 0
            subgraphs_nids_list = []
            subgraphs_eids_list = []
            for num_nodes, subgraph_nids, subgraph_eids in batch:
                sum_num_nodes += num_nodes
                subgraphs_nids_list.append(subgraph_nids)
                subgraphs_eids_list.append(subgraph_eids)
            return sum_num_nodes, subgraphs_nids_list, subgraphs_eids_list


def build_saint_dataloader(feats, graph, dataset_name, online=False, **kwargs):
    num_nodes = graph.num_nodes()
    num_subg_sampler = 1000 
    num_subg = 50 
    full = True
    batch_size_sampler = 50 

    num_roots = 10 
    
    length = 4
    num_workers = 4

    params = {
        'dn': dataset_name, 'g': graph, 'num_workers_sampler': 4,
        'num_subg_sampler': num_subg_sampler, 
        'batch_size_sampler': batch_size_sampler,
        'online': online, 'num_subg': num_subg, 
        'full': full
    }

    saint_sampler = SAINTRandomWalkLoader(feats, num_roots, length, **params)
    loader = DataLoader(saint_sampler, collate_fn=saint_sampler.__collate_fn__, batch_size=1, **kwargs)
    return loader


def get_saint_subgraphs(graph, dataset_name, online=False, num_roots=100, length=4):
    num_nodes = graph.num_nodes()
    num_subg_sampler = 20000 
    num_subg = 50 
    full = True
    batch_size_sampler = 200 

    
    
    
    num_workers = 4

    params = {
        'dn': dataset_name, 'g': graph, 'num_workers_sampler': 4,
        'num_subg_sampler': num_subg_sampler, 
        'batch_size_sampler': batch_size_sampler,
        'online': online, 'num_subg': num_subg, 
        'full': full
    }
    saint_sampler = SAINTRandomWalkLoader(None, num_roots, length, **params)
    subgraph_nodes = saint_sampler.get_generated_subgraph_nodes()
    return subgraph_nodes


if __name__ == "__main__":
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset("ogbn-arxiv", root="./data/arxiv/")
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    
    
    
    
    
    
    
    
    subgraphs = get_saint_subgraphs(g, "ogbn-arxiv")
    print(len(subgraphs))
    print(subgraphs[0].shape)
