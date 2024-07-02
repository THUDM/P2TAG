"""
This file is to pack the classes and functions for graph prompt.

credit:  The part of codes are from Sun et. al. All in One: Multi-Task Prompting for Graph Neural Networks. KDD 2023.
""" 
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProG"))
from ProG.prog_utils import seed_everything, seed

from torch import nn, optim
import torch
from copy import deepcopy

seed_everything(seed)

from ProG.prog_utils import mkdir, load_data4pretrain
from ProG.prompt import LightPrompt, HeavyPrompt, MeanAnwser, FrontAndHead
from torch import nn, optim
from ProG.data import multi_class_NIG
import torch
from torch_geometric.loader import DataLoader
from ProG.eva import acc_f1_over_batches

from random import shuffle
from ProG.meta import MAML
from ProG.eva import acc_f1_over_batches

import dgl
import time


def model_components(args, N, K): 
    adapt_lr = 0.01
    meta_lr = 0.001
    input_dim, hid_dim = args.hidden_size, args.hidden_size
    model = FrontAndHead(input_dim=input_dim, hid_dim=hid_dim, num_classes=N,  
                         task_type="multi_class_classification_mean",
                         token_num=args.token_num, cross_prune=0.1, inner_prune=0.3)

    maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=True)

    opt = optim.Adam(filter(lambda p: p.requires_grad, maml.parameters()), meta_lr)

    lossfn = nn.CrossEntropyLoss(reduction='mean')

    return maml, opt, lossfn

def model_create(num_class, args, tune_answer=True, meta=False, emb_label=None): 
    input_dim, hid_dim = args.hidden_size, args.hidden_size
    lr, wd = 0.001, 0.00001
    
    
    lossfn = nn.CrossEntropyLoss(reduction='mean')

    # if not args.label_as_init:
    #     emb_label = None
    PG = HeavyPrompt(token_dim=input_dim, token_num=args.token_num, group_num=num_class, cross_prune=0.1, inner_prune=0.3, emb_label=emb_label, cat_emb=args.LM_as_init, text_prompt=args.LM_as_init)

    opi = optim.Adam(filter(lambda p: p.requires_grad, PG.parameters()),
                        lr=lr,
                        weight_decay=wd)

    
    
    
    answering = MeanAnwser(hid_dim=hid_dim, num_class=num_class, type_=args.prompt_type, cat_emb=args.LM_as_init)

    
    
    opi_answer = optim.Adam(answering.parameters(), lr=0.01,
                            weight_decay=0.00001)
    return PG, answering, opi_answer, opi, lossfn

def meta_train_maml(epoch, maml, gnn, lossfn, opt, train_loader, adapt_steps=2, N_way=5, K_shot=100, args=None):
    for ep in range(epoch):
        meta_train_loss = 0.0
        PrintN = 10
        start_time = time.time()

        for batch, (_, batch_item) in enumerate(train_loader):
            batch_nodes, sqt_idx, qrt_idx, val_idx, sqt_graphs, qry_graphs, val_graphs = batch_item
            learner = maml.clone()

            spt_labels = torch.repeat_interleave(torch.arange(N_way), K_shot).to(gnn.device)
            qry_labels = torch.repeat_interleave(torch.arange(N_way), args.k_qry).to(gnn.device)

            
            for j in range(adapt_steps):
                support_loss = 0.0
                pre_list = []
                for support_graph in sqt_graphs:
                    support_graph = support_graph.to(f"cuda:{gnn.device}")
                    support_batch_preds = learner(support_graph, gnn)
                    pre_list.append(support_batch_preds)
                support_batch_loss = lossfn(torch.cat(pre_list), spt_labels)
                support_loss += support_batch_loss

                learner.adapt(support_loss)

            
            running_loss, query_loss = 0.0, 0.0
            pre_list = []
            for query_graph in qry_graphs:
                query_graph = query_graph.to(f"cuda:{gnn.device}")
                query_batch_preds = learner(query_graph, gnn)
                pre_list.append(query_batch_preds)

            query_batch_loss = lossfn(torch.cat(pre_list), qry_labels)
            query_loss += query_batch_loss

            
            last_loss = query_loss / len(qry_graphs)
            print('Batch {}/{} | query loss: {:.8f} | Time remaining: {:.2f}s'.format(batch + 1, len(train_loader), last_loss, (time.time() - start_time) * (len(train_loader) - batch - 1) / (batch + 1)))

            meta_train_loss += last_loss

        
        print('meta_train_loss @ epoch {}/{}: {}'.format(ep + 1, epoch, meta_train_loss.item()))

        
        opt.zero_grad()
        meta_train_loss.backward()
        opt.step()
