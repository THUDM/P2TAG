import os
import dgl
import pickle

from utils.functions import split_graph, split_time

def AmazonDataset(data_name, data_root, splits, train_ratio=0.6, val_ratio=0.2, train_year=2016, val_year=2017):
    
    graph = dgl.load_graphs(f"{data_root}{data_name}.pt")[0][0]
    
    processed_path = os.path.join(data_root, "processed")
    if not os.path.exists(processed_path):
        if splits == 'random':
            split_idx = split_graph(graph.num_nodes(), train_ratio, val_ratio)
        elif splits == 'time':
            split_idx = split_time(graph, train_year, val_year)
            del graph.ndata["year"]
        else:
            raise ValueError('Please check the way of splitting the datasets')
        os.makedirs(processed_path)
        pickle.dump(split_idx, open(os.path.join(processed_path, "split_idx.pkl"), "wb"))
        print("processed finish")
    else:
        split_idx = pickle.load(open(os.path.join(processed_path, "split_idx.pkl"), "rb"))
        
    return [(graph, split_idx)]
























































































