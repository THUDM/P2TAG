import torch
import tqdm
import random
import os
from copy import deepcopy
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.weight_norm import weight_norm
from transformers import AutoModel, AutoTokenizer
from transformers.models.deberta.modeling_deberta import DebertaOnlyMLMHead
from diffusers.models.cross_attention import CrossAttention
from functools import partial

from utils.loss_func import *
from edcoder import PreModel, setup_module
from utils.functions import euclidean_dist, aug

class BaseLM(nn.Module):
    def __init__(self, args, mask_token_id) -> None:
        super(BaseLM, self).__init__()
        self.lm_type = args.lm_type
        
        self.dropout = nn.Dropout(args.dropout)
        self.mask_rate = args.mask_rate
        self.device = args.device
        self.mask_token_id = mask_token_id

        
        self.lm_model = AutoModel.from_pretrained(self.lm_type)
        self.cls = DebertaOnlyMLMHead(self.lm_model.config)
        
        self.criterion = CrossEntropyLoss()

    def forward(self, batch, masked_input_ids):
        masked_batch = batch.copy()
        masked_batch['input_ids'] = masked_input_ids
        masked_lm_outputs = self.lm_model(**masked_batch)
        masked_emb = masked_lm_outputs.last_hidden_state
        labels = batch["input_ids"] 
        labels[masked_input_ids != self.mask_token_id] = -100
        prediction_scores = self.cls(masked_emb)
        loss = self.criterion(prediction_scores.view(-1, self.lm_model.config.vocab_size), labels.view(-1).long())
        

        return loss

    def emb(self, batch):
        with torch.no_grad():
            lm_outputs = self.lm_model(**batch)
        emb = lm_outputs.last_hidden_state
        cls_token_emb = emb.permute(1, 0, 2)[0]

        return cls_token_emb


class JointModel(nn.Module):
    def __init__(self, args, mask_token_id) -> None:
        super(JointModel, self).__init__()
        self.lm_type = args.lm_type
        self.lm_path = args.lm_path
        
        self.dropout = nn.Dropout(args.dropout)
        self.mask_rate = args.mask_rate
        self.device = args.device
        self.mask_token_id = mask_token_id
        self.lam = args.lam
        self.momentum = args.momentum
        self.delayed_ema_epoch = args.delayed_ema_epoch
        self.sup = args.sup
        self.n_way = args.n_way
        self.k_sqt = args.k_sqt
        self.k_qry = args.k_qry

        
        try:
            self.lm_model = AutoModel.from_pretrained(self.lm_type, cache_dir=self.lm_path)
        except OSError:
            self.lm_model = AutoModel.from_pretrained(os.path.expanduser("~") + "/deberta-base")
        self.cls = DebertaOnlyMLMHead(self.lm_model.config)

        
        self.gnn_encoder = setup_module(
            m_type=args.gnn_type,
            enc_dec="encoding",
            in_dim=self.lm_model.config.hidden_size,
            num_hidden=args.hidden_size // args.nhead,
            out_dim=args.hidden_size,
            num_layers=args.num_layers,
            nhead=args.nhead,
            nhead_out=1,
            concat_out=True,
            activation=args.activation,
            dropout=args.dropout,
            attn_drop=args.dropout,
            negative_slope=args.negative_slope,
            residual=True,
            norm=args.norm,
        )
        self.linear = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size),
            nn.PReLU(),
        )
        if self.lam > 0:
            self.gnn_encoder_ema = setup_module(
                m_type=args.gnn_type,
                enc_dec="encoding",
                in_dim=self.lm_model.config.hidden_size,
                num_hidden=args.hidden_size // args.nhead,
                out_dim=args.hidden_size,
                num_layers=args.num_layers,
                nhead=args.nhead,
                nhead_out=1,
                concat_out=True,
                activation=args.activation,
                dropout=args.dropout,
                attn_drop=args.dropout,
                negative_slope=args.negative_slope,
                residual=True,
                norm=args.norm,
            )
            self.gnn_encoder_ema.load_state_dict(self.gnn_encoder.state_dict())
            for p in self.gnn_encoder_ema.parameters():
                p.requires_grad = False
                p.detach_()

            self.predictor = nn.Sequential(
                nn.Linear(args.hidden_size, args.hidden_size),
                nn.PReLU(),
                nn.Linear(args.hidden_size, args.hidden_size)
            )
        self.cross_attn = CrossAttention(args.hidden_size, args.hidden_size, heads=args.nhead, dim_head=args.hidden_size // args.nhead, dropout=args.dropout)
        
        self.criterion = CrossEntropyLoss()

    def forward(self, batch, graph, epoch=0, drop_g1=None, drop_g2=None, sqt_idx=None, qrt_idx=None, new_labels=None):
        
        masked_batch = {
            "input_ids": batch["masked_input_ids"],
            "attention_mask": batch["attention_mask"],
            "token_type_ids": batch["token_type_ids"],
        }
        masked_lm_outputs = self.lm_model(**masked_batch)

        
        

        masked_emb = masked_lm_outputs.last_hidden_state
        
        masked_cls_token_emb = masked_emb.permute(1, 0, 2)[0] 
        
        

        drop_g1 = drop_g1 if drop_g1 is not None else graph
        gnn_emb = self.gnn_encoder(graph, masked_cls_token_emb)
        

        if self.lam > 0:
            with torch.no_grad():
                raw_batch = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "token_type_ids": batch["token_type_ids"],
                }
                lm_outputs = self.lm_model(**raw_batch)
                emb = lm_outputs.last_hidden_state
                cls_token_emb = emb.permute(1, 0, 2)[0]
                drop_g2 = drop_g2 if drop_g2 is not None else graph
                gnn_emb_ema = self.gnn_encoder_ema(drop_g2, cls_token_emb)

            pred = self.predictor(gnn_emb)
            loss_latent = sce_loss(pred, gnn_emb_ema, 1)
        else:
            loss_latent = torch.tensor(0)
        
        sup_loss = torch.tensor(0)
        mask_loss = torch.tensor(0)
        fw_loss = torch.tensor(0)
        graph_loss = torch.tensor(0)
        
        if self.sup:
            train_mask = graph.ndata['train_mask']
            pred = gnn_emb[train_mask]
            label = graph.ndata['label'][train_mask]
            sup_loss = F.cross_entropy(pred, label)
        else:
            '''ori'''
            shape = gnn_emb.shape
            masked_emb = torch.cat([masked_emb, gnn_emb.unsqueeze(1).expand(shape[0], batch["input_ids"].shape[1], shape[1])], dim=-1)
            hidden_emb = self.linear(masked_emb)
            '''ori'''
            '''ori'''
            labels = batch["input_ids"] 
            labels[batch["masked_input_ids"] != self.mask_token_id] = -100
            prediction_scores = self.cls(hidden_emb)
            mask_loss = self.criterion(prediction_scores.view(-1, self.lm_model.config.vocab_size), labels.view(-1).long())
            '''ori'''
        loss = sup_loss + mask_loss + self.lam * loss_latent + fw_loss + graph_loss
        if self.lam > 0 and epoch >= self.delayed_ema_epoch:
            self.update(self.gnn_encoder, self.gnn_encoder_ema)

        return loss, loss_latent

    def update(self, student, teacher):
        with torch.no_grad():
            m = self.momentum
            for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def emb(self, batch):
        with torch.no_grad():
            lm_outputs = self.lm_model(**batch)
        emb = lm_outputs.last_hidden_state
        cls_token_emb = emb.permute(1, 0, 2)[0]

        return cls_token_emb
    
    def inference(self, graph, device, batch_size=128):
        with torch.no_grad():
            output = self.gnn_encoder.inference(graph, device, batch_size)
            
        return output
    
    def inference_w_grad(self, graph, device, batch_size=128):
        for p in self.gnn_encoder.parameters():
            p.requires_grad = False
        output = self.gnn_encoder.inference(graph, device, batch_size)
        return output

