from collections import namedtuple, Counter
import numpy as np
import json
import random
import os
import pickle
from collections import defaultdict

import torch
import torch.nn.functional as F

import dgl
from dgl.data import (
    load_data, 
    TUDataset, 
)
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.dataloading import GraphDataLoader

from sklearn.preprocessing import StandardScaler
from settings import DATA_INFO
from data_tag import AmazonDataset

GRAPH_DICT = {
    "ogbn-arxiv": DglNodePropPredDataset,
    "ogbn-products": DglNodePropPredDataset,
    "Electronics-Computers": AmazonDataset,
    "Books-Children": AmazonDataset,
    "Books-History": AmazonDataset,
    "Electronics-Photo": AmazonDataset,
}



def preprocess(graph):
    ndata_backup = {k: v.clone() for k, v in graph.ndata.items()}

    if "feat" in graph.ndata:
        graph.ndata.pop("feat")

    graph = dgl.to_bidirected(graph)

    
    for key, value in ndata_backup.items():
        graph.ndata[key] = value

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name, dataset_inf, task="nc", few_shot=False):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name, root=f"./dataset/")
    elif dataset_name in ["Electronics-Computers", "Electronics-Photo", "Books-Children", "Books-History"]:
        if dataset_inf["splits"] == "time":
            assert "train_year" in dataset_inf and "val_year" in dataset_inf
            dataset = GRAPH_DICT[dataset_name](dataset_name, dataset_inf["data_root"], dataset_inf["splits"], train_year=dataset_inf["train_year"], val_year=dataset_inf["val_year"])
        elif dataset_inf["splits"] == "random":
            assert "train_ratio" in dataset_inf and "val_ratio" in dataset_inf
            dataset = GRAPH_DICT[dataset_name](dataset_name, dataset_inf["data_root"], dataset_inf["splits"], train_ratio=dataset_inf["train_ratio"], val_ratio=dataset_inf["val_ratio"])
    else:
        dataset = GRAPH_DICT[dataset_name]()
    
    if dataset_name.startswith("ogbn"):
        graph, labels = dataset[0]

        if "year" in graph.ndata:
            del graph.ndata["year"]
        if not graph.is_multigraph:
            graph = preprocess(graph)
        else:
            graph = graph.remove_self_loop().add_self_loop()

        split_idx = dataset.get_idx_split()
        if labels is not None:
            labels = labels.view(-1)
    
    elif dataset_name in ['Electronics-Computers', 'Electronics-Photo', 'Books-Children', 'Books-History']:
        graph, split_idx = dataset[0]
        labels = graph.ndata["label"]
        graph = preprocess(graph)
    
    class_split = load_class_split(dataset_name, labels, dataset_inf["data_root"]) if few_shot else None
    
    test_graph = None
    
    return graph, test_graph, labels, split_idx, class_split


def load_class_split(dataset_name, labels, dataset_path, split=True):
    if split:
        class_list_train,class_list_valid,class_list_test = json.load(open(f"{dataset_path}/split/class_split.json"))
        class_list = class_list_train + class_list_valid + class_list_test
    else:
        class_list = torch.unique(labels).tolist()
        class_list_train = []
        class_list_valid = []
        class_list_test = class_list

    class_train_dict=defaultdict(list)
    for one in class_list_train:
        for i,label in enumerate(labels.numpy().tolist()):
            if label==one:
                class_train_dict[one].append(i)
    class_valid_dict = defaultdict(list)
    for one in class_list_valid:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_valid_dict[one].append(i)


    class_test_dict = defaultdict(list)
    for one in class_list_test:
        for i, label in enumerate(labels.numpy().tolist()):
            if label == one:
                class_test_dict[one].append(i)
        
                
    class_split = {
        "train": class_train_dict,
        "valid": class_valid_dict,
        "test": class_test_dict
    }
    
    return class_split
