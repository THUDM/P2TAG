import gc
import os
import time
import logging
import os.path as osp

import dgl
import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer
import time
from settings import *
from utils.functions import sample_nodes, init_path
from tqdm import tqdm
from copy import deepcopy
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
from types import SimpleNamespace as SN


def _subset_graph(g, subset_ratio, dataset_name, sup):
    splits = ['train', 'valid', 'test']
    if subset_ratio != 1:
        subset = lambda x: x[:round(len(x) * subset_ratio)].tolist()
        split_ids = {_ + '_x': subset(sup[_ + '_x']) for _ in splits}
        seed_nodes = th.tensor(sum([split_ids[_ + '_x'] for _ in splits], []))
        node_subset = sample_nodes(g, seed_nodes, [-1])[0]
        
        g = dgl.node_subgraph(g, node_subset)
        
        
        
        
        
        
        
    else:
        split_ids = {f'{_}_x': sup[_ + '_x'] for _ in splits}
    split_len = {_: len(split_ids[f'{_}_x']) for _ in splits}
    
    logging.info(f'Loaded dataset {dataset_name} with {split_len} and {g.num_edges()} edges')
    return g, split_ids


def plot_length_distribution(node_text, tokenizer, g):
    sampled_ids = np.random.permutation(g.nodes())[:10000]
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    tokenized = tokenizer(get_text(sampled_ids), padding='do_not_pad').data['input_ids']
    node_text['text_length'] = node_text.apply(lambda x: len(x['text'].split(' ')), axis=1)
    pd.Series([len(_) for _ in tokenized]).hist(bins=20)
    import matplotlib.pyplot as plt
    plt.show()


def tokenize_graph(args):
    
    
    
    
    
    token_folder = f"{DATA_PATH}{args.dataset_name}/{args.lm_type.split('/')[-1]}/"
    if not os.path.exists(token_folder):
        init_path(token_folder)
        
        print(f'Processing data...')
        if args.dataset_name in ['products', 'papers100M', 'arxiv']:
            g_info = load_graph_info(args.dataset_name, args.task)
        print(f'Loaded graph structure, start tokenization...')
        if args.dataset_name == 'products':
            from utils.preprocess_product import _tokenize_ogb_product
            _tokenize_ogb_product(args, g_info.labels)
        elif args.dataset_name == 'papers100M':
            from utils.preprocess_paper import _tokenize_ogb_paper_datasets
            _tokenize_ogb_paper_datasets(args, g_info.labels)
        elif args.dataset_name == 'arxiv':
            _tokenize_ogb_arxiv_datasets(args, g_info.labels)
        else:
            _tokenize_textual_datasets(args)
        print(f'Tokenization finished')
    else:
        print(f'Found processed {args.dataset_name}.')


def process_graph_structure(g, cf):

    a = time.time()
    g = dgl.to_bidirected(g)
    g_info = cf.data.gi
    if cf.data.subset_ratio < 1:
        g = dgl.node_subgraph(g, cf.data.gi.IDs)
        g = g.remove_self_loop().add_self_loop()
    if cf.model in {'RevGAT','GCN'}:
        
        print(f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")
    print(f'process_graph OK!, spend {time.time() - a}')
    if 'ind' in cf.dataset:
        visible_nodes = list(set(g.nodes().numpy().tolist()) - set(g_info.splits['test_x']))
        g = dgl.node_subgraph(g, visible_nodes)
    if 'IND' in cf.dataset or 'Ind' in cf.dataset:
        test_ids = g_info.splits['test_x']
        edges_to_rm = th.cat((g.in_edges(test_ids, form='eid'), g.out_edges(test_ids, form='eid')))
        g = dgl.remove_edges(g, edges_to_rm)
        g = g.remove_self_loop().add_self_loop()
    return g  


def process_pyg_graph_structure(data, cf):
    path = '../adj_gcn.pt'
    a = time.time()
    if osp.exists(path):
        adj = th.load(path)
    else:
        N = data.num_nodes
        data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)
        row, col = data.edge_index
        print('Computing adj...')
        adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
        adj = adj.set_diag()
        deg = adj.sum(dim=1).to(th.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
        th.save(adj, path)

    adj = adj.to_scipy(layout='csr')
    print(f'process_graph OK!, spend {time.time() - a}')
    del data
    return adj


def load_ogb_graph_structure_only(dataset_name):
    from ogb.nodeproppred import DglNodePropPredDataset
    data = DglNodePropPredDataset(f"ogbn-{dataset_name}", root=init_path(OGB_ROOT))
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    return g, labels, split_idx


def load_pyg_graph_structure_only(dataset_name):
    from ogb.nodeproppred import PygNodePropPredDataset
    dataset = PygNodePropPredDataset(f"ogbn-{dataset_name}", root=init_path(OGB_ROOT))
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    return data, split_idx












def load_graph_info(dataset_name, task):
    
    
    g, labels, split_idx = load_ogb_graph_structure_only(dataset_name)
    
    splits = {**{f'{_}_x': split_idx[_].numpy() for _ in ['train', 'valid', 'test']}, 'labels': labels}
    is_gold = np.zeros((g.num_nodes()), dtype=bool)
    val_test = np.zeros((g.num_nodes()), dtype=bool)
    g, splits = _subset_graph(g, 1, dataset_name, splits)
    is_gold[splits['train_x']] = True
    val_test[splits['valid_x']] = True
    val_test[splits['test_x']] = True
    g_info = SN(splits=splits, labels=labels, is_gold=is_gold, n_nodes=g.num_nodes(), val_test=val_test)
    
    
    
    
    
    
    del g
    
    
    
            
        
        
        
        
        
        
        
    
    return g_info


def _tokenize_ogb_arxiv_datasets(args, labels, chunk_size=50000):
    def merge_by_ids(meta_data, node_ids, categories):
        meta_data.columns = ["ID", "Title", "Abstract"]
        
        meta_data["ID"] = meta_data["ID"].astype(np.int64)
        meta_data.columns = ["mag_id", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="left", on="mag_id")
        data = pd.merge(data, categories, how="left", on="label_id")
        return data

    def read_ids_and_labels(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  
        paper_ids = pd.read_csv(paper_id_path_csv)
        categories = pd.read_csv(category_path_csv)
        categories.columns = ["ID", "category"]  
        paper_ids.columns = ["ID", "mag_id"]
        categories.columns = ["label_id", "category"]
        paper_ids["label_id"] = labels
        return categories, paper_ids  

    def process_raw_text_df(meta_data, node_ids, categories):

        data = merge_by_ids(meta_data.dropna(), node_ids, categories)
        data = data[~data['title'].isnull()]
        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
        }
        
        data['text'] = data.apply(text_func[args.process_mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:args.cut_off]), axis=1)
        return data['text']

    from ogb.utils.url import download_url, extract_zip
    
    assert args.dataset_name in ['arxiv', 'papers100M']
    print(f'Loading raw text for {args.dataset_name}')
    data_info = DATA_INFO[args.dataset_name]
    token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
    info = {
        'input_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=np.uint16),
        'attention_mask': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool),
        'token_type_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool)
    }
    token_folder = f"{DATA_PATH}{args.dataset_name}/{args.lm_type.split('/')[-1]}/"
    for k, k_info in info.items():
        k_info.path = f'{token_folder}{k}.npy'
    raw_text_path = download_url(data_info['raw_text_url'], data_info['data_root'])
    print('args.lm_type', args.lm_type)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_type, cache_dir=args.lm_path)

    if args.lm_type in ['gpt2','gpt2-medium','gpt2-large','gpt2-xl']:
        print('Adding pad token')
        tokenizer.padding_side = 'left'
        tokenizer.pad_token = tokenizer.eos_token
        
    token_info = {k: np.memmap(info[k].path, dtype=info[k].type, mode='w+', shape=info[k].shape)
                  for k in token_keys}
    categories, node_ids = read_ids_and_labels(data_info['data_root'])
    for meta_data in tqdm(pd.read_table(raw_text_path, header=None, chunksize=chunk_size, skiprows=[0])):
        text = process_raw_text_df(meta_data, node_ids, categories)
        print(text.index, text)
        tokenized = tokenizer(text.tolist(), padding='max_length', truncation=True, max_length=DATA_INFO[args.dataset_name]['max_length'], return_token_type_ids=True).data
        for k in token_keys:
            token_info[k][text.index] = np.array(tokenized[k], dtype=info[k].type)
    
    return

def _tokenize_textual_datasets(args):
    
    print(f'Loading raw text for {args.dataset_name}')
    data_info = DATA_INFO[args.dataset_name]
    token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
    info = {
        'input_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=np.uint16),
        'attention_mask': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool),
        'token_type_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool)
    }
    token_folder = f"{DATA_PATH}{args.dataset_name}/{args.lm_type.split('/')[-1]}/"
    for k, k_info in info.items():
        k_info.path = f'{token_folder}{k}.npy'
    
    print('args.lm_type', args.lm_type)
    tokenizer = AutoTokenizer.from_pretrained(args.lm_type, cache_dir=args.lm_path)
    token_info = {k: np.memmap(info[k].path, dtype=info[k].type, mode='w+', shape=info[k].shape)
                  for k in token_keys}
    
    
    if args.dataset_name in ['cora_ml', 'cora', 'pubmed']:
        text_path = os.path.join(data_info["data_root"], "processed/data.txt")
    else:
        text_path = os.path.join(data_info["data_root"], f"{data_info['data_name']}.txt")
    with open(text_path, "r") as f:
        text = f.readlines()
        text = [x.strip() for x in text]
    tokenized = tokenizer(text, padding='max_length', truncation=True, max_length=DATA_INFO[args.dataset_name]['max_length'], return_token_type_ids=True).data
    for k in token_keys:
        token_info[k][np.array(range(len(text)))] = np.array(tokenized[k], dtype=info[k].type)
    return


