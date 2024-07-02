import os
import argparse
import random
import yaml
import logging
from functools import partial
import numpy as np

import dgl

import torch
import torch.nn as nn
from torch import optim as optim



logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    return False

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M


def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True


def get_current_lr(optimizer):
    return optimizer.state_dict()["param_groups"][0]["lr"]


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    graph = graph.remove_self_loop()

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src, dst = graph.edges()

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng


def build_args():
    parser = argparse.ArgumentParser(description="TAG-pretrain")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--run_entity", type=str, default="xxx")
    parser.add_argument("--dataset_name", type=str, default="arxiv")
    parser.add_argument("--lm_type", type=str, default="microsoft/deberta-base")
    parser.add_argument("--lm_path", type=str, default=None)
    parser.add_argument("--gnn_type", type=str, default="gat")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--nhead", type=int, default=4)

    parser.add_argument("--lr", type=float, default=2e-5,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="weight decay")
    parser.add_argument("--lr_f", type=float, default=0.01)
    parser.add_argument("--weight_decay_f", type=float, default=0.0)
    parser.add_argument("--negative_slope", type=float, default=0.1, help="the negative slope of leaky relu for GAT")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--mask_rate", type=float, default=0.75)
    parser.add_argument("--cut_off", type=int, default=64)
    parser.add_argument("--num_roots", type=int, default=100)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--process_mode", type=str, default="TA")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--norm", type=str, default="layernorm")
    parser.add_argument("--drop_edge_rate", type=float, default=0.0)
    # parser.add_argument("--loss_type", type=str, default="sce")

    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--lp_epochs", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--lam", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.995)
    parser.add_argument("--delayed_ema_epoch", type=int, default=0)
    parser.add_argument("--logging", action="store_true", default=False)
    
    parser.add_argument("--task", type=str, default="nc")
    parser.add_argument("--sup", action="store_true", default=False)
    
    parser.add_argument("--eval_only", type=str2bool, default=True)
    parser.add_argument("--eval_model_path", type=str, default="model_deberta-base_2.pt")
    parser.add_argument("--save_model_path", type=str, default=None)
    parser.add_argument("--few_shot", action="store_true", default=True)
    parser.add_argument("--n_way", type=str, default="5,5,10,10")
    parser.add_argument("--k_sqt", type=str, default="3,5,3,5")
    parser.add_argument("--few_shot_setting", type=str, default="5,3") # n_way, k_sqt
    parser.add_argument("--k_qry", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=5000)
    parser.add_argument("--num_repeat", type=int, default=5)
    parser.add_argument("--eval_num_tasks", type=int, default=50)

    parser.add_argument("--token_num", type=int, default=10)
    parser.add_argument("--prompt_epochs", type=int, default=10)
    parser.add_argument("--max_sample_node", type=int, default=100)
    parser.add_argument("--emb_type", type=str, default="GNN", choices=["LM", "GNN"])

    # adapt_stepus
    parser.add_argument("--meta_train", type=str2bool, default=False)
    parser.add_argument("--meta_epochs", type=int, default=1)
    parser.add_argument("--adapt_steps", type=int, default=2)

    parser.add_argument("--prompt_type", type=str, default="default", choices=["default", 
                                                                                "only_first"])

    # prompt enabling 
    parser.add_argument("--label_as_init", type=str2bool, default=True) # P_{G}
    parser.add_argument("--LM_as_init", type=str2bool, default=True) # W_{t}
    args = parser.parse_args()
    args.n_way = [int(i) for i in args.n_way.split(",")]
    args.k_sqt = [int(i) for i in args.k_sqt.split(",")]
    assert len(args.n_way) == len(args.k_sqt)
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name is None:
        return nn.Identity()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "graphnorm":
        return partial(NormLayer, norm_type="groupnorm")
    else:
        return nn.Identity


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()

    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]
    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "radam":
        optimizer = optim.RAdam(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        assert False and "Invalid optimizer"

    return optimizer


# -------------------
def aug(graph, edge_mask_rate):
    n_node = graph.num_nodes()

    edge_mask = mask_edge(graph, edge_mask_rate)
    # feat = drop_feature(x, feat_drop_rate)

    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    return ng


def drop_feature(x, drop_prob):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0

    return x


def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.FloatTensor(np.ones(E) * mask_prob)
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx


def drop_edge(graph, drop_rate, return_edges=False):
    if drop_rate <= 0:
        return graph

    n_node = graph.num_nodes()
    edge_mask = mask_edge(graph, drop_rate)
    src = graph.edges()[0]
    dst = graph.edges()[1]

    nsrc = src[edge_mask]
    ndst = dst[edge_mask]

    ng = dgl.graph((nsrc, ndst), num_nodes=n_node)
    ng = ng.add_self_loop()

    dsrc = src[~edge_mask]
    ddst = dst[~edge_mask]

    if return_edges:
        return ng, (dsrc, ddst)
    return ng


def load_best_configs(args, path):
    with open(path, "r") as f:
        configs = yaml.load(f, yaml.FullLoader)

    if args.dataset not in configs:
        logging.info("Best args not found")
        return args

    logging.info("Using best configs")
    configs = configs[args.dataset]

    for k, v in configs.items():
        if "lr" in k or "weight_decay" in k:
            v = float(v)
        setattr(args, k, v)
    print("------ Use best configs ------")
    return args


# ------ logging ------

# class TBLogger(object):
#     def __init__(self, log_path="./logging_data", name="run"):
#         super(TBLogger, self).__init__()

#         if not os.path.exists(log_path):
#             os.makedirs(log_path, exist_ok=True)

#         self.last_step = 0
#         self.log_path = log_path
#         raw_name = os.path.join(log_path, name)
#         name = raw_name
#         for i in range(1000):
#             name = raw_name + str(f"_{i}")
#             if not os.path.exists(name):
#                 break
#         self.writer = SummaryWriter(logdir=name)

#     def note(self, metrics, step=None):
#         if step is None:
#             step = self.last_step
#         for key, value in metrics.items():
#             self.writer.add_scalar(key, value, step)
#         self.last_step = step

#     def finish(self):
#         self.writer.close()


class NormLayer(nn.Module):
    def __init__(self, hidden_dim, norm_type):
        super().__init__()
        if norm_type == "batchnorm":
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == "layernorm":
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm_type == "graphnorm":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))

            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        else:
            raise NotImplementedError
        
    def forward(self, graph, x):
        tensor = x
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        batch_list = graph.batch_num_nodes
        batch_size = len(batch_list)
        batch_list = torch.Tensor(batch_list).long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias

# os util ==============================
def init_random_state(seed=0):
    # Libraries using GPU should be imported after specifying GPU-ID
    import torch
    import random
    # import dgl
    # dgl.seed(seed)
    # dgl.random.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dir_of_file(f_name):
    return os.path.dirname(f_name) + '/'


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    import errno
    if os.path.exists(path): return
    # print(path)
    # path = path.replace('\ ',' ')
    # print(path)
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def init_path(dir_or_file):
    path = get_dir_of_file(dir_or_file)
    if not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file


# dgl utils ===============================
 
def sample_nodes(g, seed_nodes, fanout_list):
    # seed_nodes = th.tensor(seed_nodes).to(g.device) if isinstance(seed_nodes, int) else seed_nodes
    induced_nodes = {0: (cur_nodes := seed_nodes.view(-1))}
    init_random_state(0)
    for l, fanout in enumerate(fanout_list):
        frontier = dgl.sampling.sample_neighbors(g, cur_nodes, fanout)
        cur_nodes = frontier.edges()[0].unique()
        induced_nodes[l + 1] = cur_nodes
    sampled_nodes = torch.cat(list(induced_nodes.values())).unique()
    return sampled_nodes, induced_nodes


def get_edge_set(g: dgl.DGLGraph):
    """graph_edge_to list of (row_id, col_id) tuple
    """

    return set(map(tuple, np.column_stack([_.cpu().numpy() for _ in g.edges()]).tolist()))


def edge_set_to_inds(edge_set):
    """ Unpack edge set to row_ids, col_ids"""
    return list(map(list, zip(*edge_set)))


def l2_normalize(x):
    norm = x.pow(2).sum(1, keepdim=True).pow(1. / 2)
    out = x.div(norm+1e-10)
    return out


# graph utils
def split_graph(nodes_num, train_ratio, val_ratio):
    np.random.seed(42)
    indices = np.random.permutation(nodes_num)
    train_size = int(nodes_num * train_ratio)
    val_size = int(nodes_num * val_ratio)

    train_ids = indices[:train_size]
    val_ids = indices[train_size:train_size + val_size]
    test_ids = indices[train_size + val_size:]
    
    split_idx = {
            "train": torch.tensor(train_ids),
            "valid": torch.tensor(val_ids),
            "test": torch.tensor(test_ids)
        }

    return split_idx


def split_time(g, train_year=2016, val_year=2017):
    np.random.seed(42)
    year = list(np.array(g.ndata['year']))
    indices = np.arange(g.num_nodes())
    # 1999-2014 train
    # Filter out nodes with label -1
    print(f'train year: {train_year}')
    valid_indices = [i for i in indices if g.ndata['label'][i] != -1]

    # Filter out valid indices based on years
    train_ids = [i for i in valid_indices if year[i] < train_year]
    val_ids = [i for i in valid_indices if year[i] >= train_year and year[i] < val_year]
    test_ids = [i for i in valid_indices if year[i] >= val_year]


    train_length = len(train_ids)
    val_length = len(val_ids)
    test_length = len(test_ids)

    print("Train set length:", train_length)
    print("Validation set length:", val_length)
    print("Test set length:", test_length)
    
    split_idx = {
            "train": torch.tensor(train_ids),
            "valid": torch.tensor(val_ids),
            "test": torch.tensor(test_ids)
        }

    return split_idx