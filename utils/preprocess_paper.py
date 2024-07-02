import gc
import os
import time

import dgl
import numpy as np
import pandas as pd
import torch as th
from transformers import AutoTokenizer


from utils.functions import sample_nodes, init_random_state, init_path
from settings import *
from tqdm import tqdm
from types import SimpleNamespace as SN


def _subset_graph(g, cf, sup):
    splits = ['train', 'valid', 'test']
    if cf.data.subset_ratio != 1:
        init_random_state(0)
        subset = lambda x: x[:round(len(x) * cf.data.subset_ratio)].tolist()
        split_ids = {_: subset(sup[_ + '_x']) for _ in splits}
        seed_nodes = th.tensor(sum([split_ids[_] for _ in splits], []))
        node_subset = sample_nodes(g, seed_nodes, [-1])[0]
        
        g = dgl.node_subgraph(g, node_subset)
    else:
        split_ids = {_: sup[_ + '_x'] for _ in splits}
    split_len = {_: len(split_ids[_]) for _ in splits}
    log_func = cf.log if hasattr(cf, 'log') else print
    log_func(f'Loaded dataset {cf.dataset} with {split_len} and {g.num_edges()} edges')
    return g, split_ids


def plot_length_distribution(node_text, tokenizer, g):
    sampled_ids = np.random.permutation(g.nodes())[:10000]
    get_text = lambda n: node_text.iloc[n]['text'].tolist()
    tokenized = tokenizer(get_text(sampled_ids), padding='do_not_pad').data['input_ids']
    node_text['text_length'] = node_text.apply(lambda x: len(x['text'].split(' ')), axis=1)
    pd.Series([len(_) for _ in tokenized]).hist(bins=20)
    import matplotlib.pyplot as plt
    plt.show()


def load_graph_and_supervision(args):
    from ogb.nodeproppred import DglNodePropPredDataset
    
    
    data = DglNodePropPredDataset(f"ogbn-{args.dataset_name}", root=init_path(OGB_ROOT))
    g, labels = data[0]
    split_idx = data.get_idx_split()
    labels = labels.squeeze().numpy()
    print('data OK!')
    
    sup = {**{f'{_}_x': split_idx[_].numpy() for _ in ["train", 'valid', 'test']}, 'labels': labels}
    
    
    
    

    if args.dataset_name not in {'ogbn-papers100M'}:
        g = dgl.to_bidirected(g)

    if args.gnn_type in {'RevGAT'}:
        
        print(f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
        g = g.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {g.number_of_edges()}")

    return g, sup


def _tokenize_ogb_paper_datasets(args, labels, chunk_size=2000000):
    def merge_by_ids(meta_data, node_ids, categories):
        meta_data.columns = ["ID", "Title", "Abstract"]
        
        meta_data["ID"] = meta_data["ID"].astype(np.int64)
        meta_data.columns = ["ID", "title", "abstract"]
        data = pd.merge(node_ids, meta_data, how="right", on="ID")  
        
        
        return data

    def merge_tit_abs(title, abs):
        title.columns = ["ID", "Title"]
        title["ID"] = title["ID"].astype(np.int64)
        abs.columns = ["ID", "Abstract"]
        abs["ID"] = abs["ID"].astype(np.int64)
        data = pd.merge(title, abs, how="outer", on="ID", sort=True)
        data.to_csv(f'{DATA_INFO[args.dataset_name]["data_root"]}titleabs.tsv', sep="\t", header=True, index=False)
        import gc
        del data
        gc.collect()
        return

    def read_ids_and_labels(data_root):
        category_path_csv = f"{data_root}/mapping/labelidx2arxivcategeory.csv.gz"
        paper_id_path_csv = f"{data_root}/mapping/nodeidx2paperid.csv.gz"  
        paper_ids = pd.read_csv(paper_id_path_csv, usecols=[0])
        categories = pd.read_csv(category_path_csv)
        categories.columns = ["ID", "category"]  
        paper_ids.columns = ["ID"]
        categories.columns = ["label_id", "category"]
        paper_ids["label_id"] = labels  
        return categories, paper_ids  

    def process_raw_text_df(meta_data, node_ids, categories):
        b = time.time()
        data = merge_by_ids(meta_data, node_ids, categories)
        print(f'waste {time.time() - b} in merge_by_ids')

        text_func = {
            'TA': lambda x: f"Title: {x['title']}. Abstract: {x['abstract']}",
            'T': lambda x: x['title'],
        }
        
        data['text'] = data.apply(text_func[args.process_mode], axis=1)
        data['text'] = data.apply(lambda x: ' '.join(x['text'].split(' ')[:512]), axis=1) 
        return data

    from ogb.utils.url import download_url, extract_zip
    
    assert args.dataset_name in ['papers100M']
    print(f'Loading raw text for {args.dataset_name}')
    if not os.path.exists(f'{DATA_INFO[args.dataset_name]["data_root"]}titleabs.tsv'):
        if not os.path.exists(f'{DATA_INFO[args.dataset_name]["data_root"]}paperinfo/idx_abs.tsv'):  
            raw_text_path = download_url(DATA_INFO[args.dataset_name]["raw_text_url"], DATA_INFO[args.dataset_name]["data_root"])
            extract_zip(raw_text_path, DATA_INFO[args.dataset_name]["data_root"])
        print('start read abstract')
        abstract = pd.read_csv(f'{DATA_INFO[args.dataset_name]["data_root"]}paperinfo/idx_abs.tsv', sep='\t', header=None)
        print('abstract ok')
        title = pd.read_csv(f'{DATA_INFO[args.dataset_name]["data_root"]}paperinfo/idx_title.tsv', sep='\t', header=None)
        print('begin merge')
        merge_tit_abs(title, abstract)
        print('merge ok')

    tokenizer = AutoTokenizer.from_pretrained(args.lm_type, cache_dir=args.lm_path)
    
    data_info =  DATA_INFO[args.dataset_name]
    info = {
        'input_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=np.uint16),
        'attention_mask': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool),
        'token_type_ids': SN(shape=(data_info['n_nodes'], data_info['max_length']), type=bool)
    }
    token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
    token_folder = f"{DATA_PATH}{args.dataset_name}/{args.lm_type.split('/')[-1]}/"
    for k, k_info in info.items():
        k_info.path = f'{token_folder}{k}.npy'
    
    token_info = {k: np.memmap(info[k].path, dtype=info[k].type, mode='w+', shape=info[k].shape)
                  for k in token_keys}
    s = time.time()
    categories, node_ids = read_ids_and_labels(DATA_INFO[args.dataset_name]["data_root"])
    print(f'waste {time.time()-s} in read_ids_and_labels')
    raw_text_path = f'{DATA_INFO[args.dataset_name]["data_root"]}titleabs.tsv'
    for meta_data in tqdm(pd.read_table(raw_text_path, chunksize=chunk_size)):  
        a = time.time()
        text = process_raw_text_df(meta_data, node_ids, categories)
        print(f'waste {time.time() - a} in process_raw_text_df')
        tokenized = tokenizer(text['text'].tolist(), padding='max_length', truncation=True, max_length=512).data
        for k in token_keys:
            token_info[k][text['ID']] = np.array(tokenized[k], dtype=info[k].type)
    
    return
