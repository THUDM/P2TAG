from types import SimpleNamespace as SN
import numpy as np
import torch
import random
import itertools
from scipy.sparse import csr_matrix

from settings import *
from utils.preprocess import *
from utils.data_util import load_dataset

from saint_sampler import get_saint_subgraphs
from fw_sampler import get_fw_graphs, get_fw_graphs_with_valid



class TAG():
    def __init__(self, args) -> None:
        self.name = args.dataset_name
        self.data_info =  DATA_INFO[self.name]
        self.data = {}
        self.lm_type = args.lm_type
        self.token_folder = f"{DATA_PATH}{self.name}/{self.lm_type.split('/')[-1]}/"
        self.n_nodes = self.data_info['n_nodes']
        self.token_keys = ['attention_mask', 'input_ids', 'token_type_ids']
        self.mask_rate = args.mask_rate
        self.device = args.device
        self.args = args

        self.graph, self.test_graph, self.labels, self.split_idx, self.class_split = load_dataset(self.data_info["data_name"], self.data_info, args.task, args.few_shot, )
        
        

        self.info = {
            'input_ids': SN(shape=(self.data_info['n_nodes'], self.data_info['max_length']), type=np.uint16),
            'attention_mask': SN(shape=(self.data_info['n_nodes'], self.data_info['max_length']), type=bool),
            'token_type_ids': SN(shape=(self.data_info['n_nodes'], self.data_info['max_length']), type=bool)
        }
        for k, k_info in self.info.items():
            k_info.path = f'{self.token_folder}{k}.npy'

        self.load_data()

    def load_data_fields(self, k_list):
        for k in k_list:
            i = self.info[k]
            try:
                self.data[k] = np.memmap(i.path, mode='r', dtype=i.type, shape=i.shape)
                self.data[k] = self.data[k][:, :self.args.cut_off]
            except:
                raise ValueError(f'Shape not match {i.shape}')

    def get_tokens(self, node_id):
        _load = lambda k: torch.IntTensor(np.array(self.data[k][node_id]))
        item = {}
        item['attention_mask'] = _load('attention_mask')
        item['input_ids'] = torch.IntTensor(np.array(self.data['input_ids'][node_id]).astype(np.int32))
        if self.lm_type not in ['distilbert-base-uncased','roberta-base']:
            item['token_type_ids'] = _load('token_type_ids')
        return item

    def load_data(self):
        tokenize_graph(self.args)
        self.load_data_fields(self.token_keys)
        self.get_end_index()

    def get_end_index(self):
        self.token_length_list = []
        for i in range(self.n_nodes):
            zero_index = torch.where(torch.IntTensor(np.array(self.data['input_ids'][i]).astype(np.int32))==0)[0]
            if len(zero_index) == 0:
                end_index = len(self.data['input_ids'][i]) - 1
            else:
                end_index = int(zero_index[0] - 1)
            token_length = end_index 
            self.token_length_list.append(token_length)


class TAGDataset(torch.utils.data.Dataset):  
    def __init__(self, data: TAG, lm_path):
        self.data = data
        
        tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type, cache_dir=lm_path)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]

    def __getitem__(self, node_id):
        item = self.data.get_tokens(node_id)
        masked_input_ids = self.get_masked_item(item, self.data.token_length_list[node_id])
        return item, masked_input_ids

    def get_masked_item(self, item, token_length):
        if token_length <= 0:
            masked_input_ids = item['input_ids'].clone()
        else:
            mask_num = int(token_length * self.data.mask_rate)
            
            mask_list = random.sample(range(1, token_length + 1), mask_num)
            masked_input_ids = item['input_ids'].index_fill(0, torch.tensor(mask_list, dtype=torch.long), self.mask_token_id)
        return masked_input_ids


    def __len__(self):
        return self.data.n_nodes

class IterTAGDataset(torch.utils.data.IterableDataset):  
    def __init__(self, data: TAG, lm_path=None, batch_size=None, num_roots=100, length=4, mode="train", num_repeat=1, num_tasks=50, N=0, K=0, Q=0):
        self.data = data
        tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type, cache_dir=lm_path)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
        self.N = N
        self.K = K
        self.Q = Q
        self.num_repeat = 1 if mode == "train" else num_repeat
        self.current_round = 0 

        self.loader = get_saint_subgraphs(self.data.graph, self.data.data_info["data_name"], num_roots=num_roots, length=length)
        self.loader = [self.loader]

        self.batch_size = batch_size
        self.mode = mode

    def __getitem__(self, node_id):
        item = self.data.get_tokens(node_id)
        masked_input_ids = self.get_masked_item(item, self.data.token_length_list[node_id])
        return item, masked_input_ids

    def get_masked_item(self, item, token_length):
        if token_length <= 0:
            masked_input_ids = item['input_ids'].clone()
        else:
            mask_num = int(token_length * self.data.mask_rate)
            
            mask_list = random.sample(range(1, token_length + 1), mask_num)
            masked_input_ids = item['input_ids'].index_fill(0, torch.tensor(mask_list, dtype=torch.long), self.mask_token_id)
        return masked_input_ids

    def __iter__(self):
        loader = self.loader
        for iter_data in zip(*loader):
            iter_data = iter_data[0]
            if self.batch_size is not None:
                iter_data = iter_data[:self.batch_size]
            
            batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            batch["masked_input_ids"] = []
            
            for key in iter_data:
                item, masked_input_ids = self.__getitem__(key)
                batch["input_ids"].append(item["input_ids"])
                batch["attention_mask"].append(item["attention_mask"])
                batch["token_type_ids"].append(item["token_type_ids"])
                batch["masked_input_ids"].append(masked_input_ids)
                
            batch["input_ids"] = torch.stack(batch["input_ids"], 0)
            batch["attention_mask"] = torch.stack(batch["attention_mask"], 0)
            batch["token_type_ids"] = torch.stack(batch["token_type_ids"], 0)
            batch["masked_input_ids"] = torch.stack(batch["masked_input_ids"], 0)
            
            yield batch, iter_data

    def __len__(self):
        return len(self.loader[0])
        
    
class IterFewShotDataset(torch.utils.data.IterableDataset):  
    def __init__(self, data: TAG, mode="train", num_repeat=1, num_tasks=50, N=0, K=0, Q=0, emb=None, model=None, need_val=True, is_graph=False, max_sample_node=1000, woprompt=False):
        self.data = data
        self.max_sample_node = max_sample_node
        self.N = N
        self.K = K
        self.Q = Q
        self.num_repeat = 1 if mode == "train" else num_repeat
        self.current_round = 0 
        self.data.graph = self.data.graph.to(self.data.device)
        self.woprompt = woprompt
        if is_graph:
            self.data.graph.ndata['feat'] = emb

        # self.loader = get_fw_graphs(mode, self.data.class_split, self.data.data_info["data_name"], N, K, Q, num_tasks, self.num_repeat)
        
        self.need_val = need_val
        if self.need_val:
            self.loader = get_fw_graphs_with_valid(mode, self.data.class_split, self.data.data_info["data_name"], N, K, Q, num_tasks, self.num_repeat)
        else:
            self.loader = get_fw_graphs(mode, self.data.class_split, self.data.data_info["data_name"], N, K, Q, num_tasks, self.num_repeat)

        if is_graph:
            num_nodes = self.data.graph.number_of_nodes()
            src, dst = self.data.graph.edges()
            adj_matrix_csr = csr_matrix((torch.ones(len(src.cpu())), (src.cpu(), dst.cpu())), shape=(num_nodes, num_nodes))
            self.dic_neis = {node: adj_matrix_csr.indices[adj_matrix_csr.indptr[node]:adj_matrix_csr.indptr[node + 1]].tolist() 
                            for node in range(num_nodes)}

        self.num_tasks = num_tasks 
        
        self.mode = mode
        self.is_graph = is_graph
        
        self.embed_label = True if model else False
        
        if self.embed_label:
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type, cache_dir=self.data.args.lm_path)
            
            
            self.label_emb = self.get_label_emb(model)
        
    def tokenize(self, texts):
        tokenized = self.tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=self.data.data_info['max_length'], return_token_type_ids=True)
        return tokenized
    
    def get_label_emb(self, model):
        if self.data.data_info["data_name"].startswith("ogbn"):
            if "arxiv" in self.data.data_info["data_name"]:
                middle = "labelidx2arxivcategeory.csv.gz"
            else:
                middle = "labelidx2productcategory.csv.gz"
            category_path_csv = f"{self.data.data_info['data_root']}/mapping/{middle}"
            categories = pd.read_csv(category_path_csv)
            categories = categories.fillna("the blank category")
            categories.columns = ["ID", "category"]  
            categories.columns = ["label_id", "category"]
            labels_texts = categories['category'].to_numpy()
        else:
            category_path = f"{self.data.data_info['data_root']}/labelid_text.txt"
            with open(category_path, 'r') as f:
                lines = f.readlines()
                labels_texts = np.array([x.strip() for x in lines])
        
        unique_labels_texts = np.unique(labels_texts)
        
        batchs = self.tokenize(unique_labels_texts)
        batchs = {k: torch.from_numpy(np.array(v)).to(self.data.device) for k, v in batchs.items()}
        unique_label_emb = model.emb(batchs)
        
        label2emb = {label: unique_label_emb[i] for i, label in enumerate(unique_labels_texts)}
        
        label_emb = torch.stack([label2emb[label] for label in labels_texts])
        return label_emb
        
        # batchs = self.tokenize(labels_texts)
        # batchs = {k: torch.from_numpy(np.array(v)).to(self.data.device) for k, v in batchs.items()}
        # label_emb = model.emb(batchs)
        # return label_emb

    def iter(self, iter_data):
        sqt_nodes, qry_nodes, classes_repeat = iter_data
        sqt_nodes = torch.tensor(sqt_nodes)
        qry_nodes = torch.tensor(qry_nodes)
        
        iter_data, inverse_indices = torch.unique(torch.cat((sqt_nodes, qry_nodes)), return_inverse=True)
        sqt_idx, qrt_idx = inverse_indices.split([sqt_nodes.size(0), qry_nodes.size(0)])
        if self.is_graph:
            sqt_graphs = self.sample_neighbor_graph_by_center_id_list(sqt_nodes, max_sample_node=self.max_sample_node)
            qry_graphs = self.sample_neighbor_graph_by_center_id_list(qry_nodes, max_sample_node=self.max_sample_node)

        batch = {}
        if self.embed_label:
            emb_classes = self.label_emb[classes_repeat]
            return batch, (iter_data, sqt_idx, qrt_idx, sqt_graphs, qry_graphs, emb_classes)
        else:
            return batch, (iter_data, sqt_idx, qrt_idx, sqt_graphs, qry_graphs)
                
    def iter_with_valid(self, iter_data):
        sqt_nodes, qry_nodes, val_nodes, classes_repeat = iter_data
        sqt_nodes = torch.tensor(sqt_nodes)
        qry_nodes = torch.tensor(qry_nodes)
        val_nodes = torch.tensor(val_nodes)
        
        iter_data, inverse_indices = torch.unique(torch.cat((sqt_nodes, qry_nodes, val_nodes)), return_inverse=True)
        sqt_idx, qrt_idx, val_idx = inverse_indices.split([sqt_nodes.size(0), qry_nodes.size(0), val_nodes.size(0)])
        
        if self.is_graph:
            sqt_graphs = self.sample_neighbor_graph_by_center_id_list(sqt_nodes, max_sample_node=self.max_sample_node)
            qry_graphs = self.sample_neighbor_graph_by_center_id_list(qry_nodes, max_sample_node=self.max_sample_node)
            val_graphs = self.sample_neighbor_graph_by_center_id_list(val_nodes, max_sample_node=self.max_sample_node)

        batch = {}
        if self.woprompt:
            return batch, (iter_data, sqt_idx, val_idx, qrt_idx)
        else:
            if self.embed_label:
                emb_classes = self.label_emb[classes_repeat]
            else:
                emb_classes = None
            return batch, (iter_data, sqt_idx, val_idx, qrt_idx, sqt_graphs, val_graphs, qry_graphs, emb_classes)
        
    def __iter__(self):
        loader = [self.loader[x][self.current_round] for x in range(len(self.loader))]
        for iter_data in zip(*loader):
            if self.need_val:
                batch, batch_item = self.iter_with_valid(iter_data)
            else:
                batch, batch_item = self.iter(iter_data)
            yield batch, batch_item

    def sample_neighbor_graph_by_center_id_list(self, center_id_list, max_sample_node=10, h=1):
        if isinstance(center_id_list, torch.Tensor):
            center_id_list = center_id_list.tolist()
        elif not isinstance(center_id_list, list):
            center_id_list = [center_id_list]
        graph = self.data.graph
        subgraph_list = []
        for center_id in center_id_list:
            neighborhood = self.dic_neis[center_id]  
            if h == 2:
                neighborhood2 = [self.dic_neis[node] for node in neighborhood]
                neighborhood = list(itertools.chain(*neighborhood2)) + neighborhood
            neighborhood = set(neighborhood)
            
            
            if center_id in neighborhood:
                neighborhood.remove(center_id)
            neighborhood = list(neighborhood)
            
            if len(neighborhood) >= max_sample_node:
                neighborhood = random.sample(neighborhood, max_sample_node - 1)
            
            
            neighborhood.insert(0, center_id)
            
            subgraph = graph.subgraph(neighborhood, store_ids=True)
            subgraph.ndata['feat'] = graph.ndata['feat'][neighborhood]
            
            subgraph_list.append(subgraph)
        return subgraph_list

    def __len__(self):
        return self.num_tasks


class IterLabeledTAGDataset(torch.utils.data.IterableDataset):  
    def __init__(self, data: TAG, batch_size=None, num_roots=100, length=4):
        self.data = data
        tokenizer = AutoTokenizer.from_pretrained(self.data.lm_type)
        self.mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
        self.loader = get_saint_subgraphs(self.data.graph, self.data.data_info["data_name"], num_roots=num_roots, length=length)
        self.batch_size = batch_size

    def __getitem__(self, node_id):
        item = self.data.get_tokens(node_id)
        masked_input_ids = self.get_masked_item(item, self.data.token_length_list[node_id])
        return item, masked_input_ids

    def get_masked_item(self, item, token_length):
        if token_length <= 0:
            masked_input_ids = item['input_ids'].clone()
        else:
            mask_num = int(token_length * self.data.mask_rate)
            mask_list = random.sample(range(1, token_length + 1), mask_num)
            masked_input_ids = item['input_ids'].index_fill(0, torch.tensor(mask_list, dtype=torch.long), self.mask_token_id)
        return masked_input_ids

    def __iter__(self):
        for iter_data in self.loader:
            
            if self.batch_size is not None:
                iter_data = iter_data[:self.batch_size]
            batch = {"input_ids": [], "attention_mask": [], "token_type_ids": []}
            batch["masked_input_ids"] = []
            
            for key in iter_data:
                item, masked_input_ids = self.__getitem__(key)
                batch["input_ids"].append(item["input_ids"])
                batch["attention_mask"].append(item["attention_mask"])
                batch["token_type_ids"].append(item["token_type_ids"])
                batch["masked_input_ids"].append(masked_input_ids)

            batch["input_ids"] = torch.stack(batch["input_ids"], 0)
            batch["attention_mask"] = torch.stack(batch["attention_mask"], 0)
            batch["token_type_ids"] = torch.stack(batch["token_type_ids"], 0)
            batch["masked_input_ids"] = torch.stack(batch["masked_input_ids"], 0)

            yield batch, iter_data

    def __len__(self):
        return len(self.loader)