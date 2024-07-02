import copy
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim as optim
import numpy as np
import random

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

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

def node_classification_evaluation(graph, x, labels, split_idx, num_classes, lr_f, weight_decay_f, max_epoch_f, device, mute=False):
    in_feat = x.shape[1]
    encoder = LogisticRegression(in_feat, num_classes)

    num_finetune_params = [p.numel() for p in encoder.parameters() if  p.requires_grad]
    if not mute:
        print(f"num parameters for finetuning: {sum(num_finetune_params)}")
    
    encoder.to(device)
    optimizer_f = create_optimizer("adam", encoder, lr_f, weight_decay_f)
    final_acc, estp_acc, best_val_acc = linear_probing_for_transductive_node_classification(encoder, graph, x, labels, split_idx, optimizer_f, max_epoch_f, device, mute)
    return final_acc, estp_acc, best_val_acc


def linear_probing_for_transductive_node_classification(model, graph, feat, labels, split_idx, optimizer, max_epoch, device, mute=False):
    criterion = torch.nn.CrossEntropyLoss()

    graph = graph.to(device)
    x = feat.to(device)
    labels = labels.to(device)

    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
    if not torch.is_tensor(train_idx):
        train_idx = torch.as_tensor(train_idx)
        val_idx = torch.as_tensor(val_idx)
        test_idx = torch.as_tensor(test_idx)

    num_nodes = graph.num_nodes()
    train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True).to(device)
    val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True).to(device)
    test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True).to(device)
    
    
    
    

    best_val_acc = 0
    best_val_epoch = 0
    best_model = None

    if not mute:
        epoch_iter = tqdm(range(max_epoch))
    else:
        epoch_iter = range(max_epoch)

    for epoch in epoch_iter:
        model.train()
        out = model(graph, x)
        loss = criterion(out[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()

        with torch.no_grad():
            model.eval()
            pred = model(graph, x)
            val_acc = accuracy(pred[val_mask], labels[val_mask])
            val_loss = criterion(pred[val_mask], labels[val_mask])
            test_acc = accuracy(pred[test_mask], labels[test_mask])
            test_loss = criterion(pred[test_mask], labels[test_mask])
        
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch
            best_model = copy.deepcopy(model)


    best_model.eval()
    with torch.no_grad():
        pred = best_model(graph, x)
        estp_test_acc = accuracy(pred[test_mask], labels[test_mask])
    print(f"--- TestAcc: {test_acc:.4f}, early-stopping-TestAcc: {estp_test_acc:.4f}, Best ValAcc: {best_val_acc:.4f} in epoch {best_val_epoch} --- ")

    
    return test_acc, estp_test_acc, best_val_acc

class LogisticRegression(nn.Module):
    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)

    def forward(self, g, x, *args):
        logits = self.linear(x)
        return logits


def link_prediction_evaluation(graph, x):
    
    
    
    
    
    
    
    node2vec = {i: row for i, row in enumerate(x)}
    
    edge_src, edge_dst = graph.edges()
    edge_index = torch.stack((edge_src, edge_dst), dim=0).numpy()
    edges = edge_index.transpose()
    nodes = list(set([i for j in edges for i in j]))
    a = 0
    b = 0
    for i, j in edges:
        if i in node2vec.keys() and j in node2vec.keys():
            dot1 = np.dot(node2vec[i], node2vec[j])
            random_node = random.sample(nodes, 1)[0]
            while random_node == j or random_node not in node2vec.keys():
                random_node = random.sample(nodes, 1)[0]
            dot2 = np.dot(node2vec[i], node2vec[random_node])
            if dot1 > dot2:
                a += 1
            elif dot1 == dot2:
                a += 0.5
            b += 1
    print("")
    print("Auc value:", float(a) / b)