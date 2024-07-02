import torch
import torch.nn.functional as F
import dgl
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, TransformerConv
from torch_geometric.data import Batch, Data
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ProG"))
from prog_utils import act
import warnings

class MeanAnwser(torch.nn.Module):
    def __init__(self, hid_dim, num_class, type_="default", cat_emb=True):
        super(MeanAnwser, self).__init__()
        self.type_ = type_
        self.cat_emb = cat_emb
        if cat_emb:
            self.linear = torch.nn.Linear(hid_dim*2, num_class)
        else:
            self.linear = torch.nn.Linear(hid_dim, num_class)
        self.softmax = torch.nn.Softmax(dim=1) 
    
    def forward_default(self, batched_graph, x=None, emb=None, emb_label=None):
        if x is not None:
            batched_graph.ndata['h'] = x

        mean_tensor = dgl.mean_nodes(batched_graph, 'h')
        if self.cat_emb:
            mean_tensor = torch.cat([mean_tensor, emb.to(mean_tensor.device)], dim=1)
        return self.linear(mean_tensor)
    
    def forward_only_first(self, batched_graph, x=None, emb=None, emb_label=None):
        if x is not None:
            batched_graph.ndata['h'] = x

        batch_num_nodes = batched_graph.batch_num_nodes().tolist() 
        first_node_indices = [sum(batch_num_nodes[:i]) for i in range(len(batch_num_nodes))]
        last_node_indices = [first_node_indices[i] + batch_num_nodes[i] - 1 for i in range(len(batch_num_nodes))]
        last_node_feature= batched_graph.ndata['h'][last_node_indices]

        mean_tensor = last_node_feature
        if self.cat_emb:
            mean_tensor = torch.cat([mean_tensor, emb.to(mean_tensor.device)], dim=1)
        return self.linear(mean_tensor)

    def forward(self, *args, **kwargs):
        if self.type_ == "default":
            return self.forward_default(*args, **kwargs)
        elif self.type_ == "only_first":
            return self.forward_only_first(*args, **kwargs)
        else:
            raise NotImplementedError


class LightPrompt(torch.nn.Module):
    def __init__(self, token_dim, token_num_per_group, group_num=1, inner_prune=None, emb_label=None, cat_emb=False, text_prompt=False):
        """
        :param token_dim:
        :param token_num_per_group:
        :param group_num:   the total token number = token_num_per_group*group_num, in most cases, we let group_num=1.
                            In prompt_w_o_h mode for classification, we can let each class correspond to one group.
                            You can also assign each group as a prompt batch in some cases.

        :param prune_thre: if inner_prune is None, then all inner and cross prune will adopt this prune_thre
        :param isolate_tokens: if Trure, then inner tokens have no connection.
        :param inner_prune: if inner_prune is not None, then cross prune adopt prune_thre whereas inner prune adopt inner_prune
        """
        super(LightPrompt, self).__init__()

        self.token_num_per_group = token_num_per_group
        self.inner_prune = inner_prune

        self.token_list = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.empty(token_num_per_group, token_dim)) for i in range(group_num)])

        self.token_init(init_method="kaiming_uniform", emb_label=emb_label)

        if text_prompt:
            self.text_prompt = torch.nn.Linear(token_dim, token_dim, bias=False) 
            


    def token_init(self, init_method="kaiming_uniform", emb_label=None):
        if init_method == "kaiming_uniform":
            for token in self.token_list:
                torch.nn.init.kaiming_uniform_(token, nonlinearity='leaky_relu', mode='fan_in', a=0.01)
        else:
            raise ValueError("only support kaiming_uniform init, more init methods will be included soon")
        if emb_label is not None:
            if len(emb_label) > self.token_num_per_group:
                warnings.warn(f"len(emb_label): {len(emb_label)} is larger than self.token_num_per_group: {self.token_num_per_group}, consider to increase .")
            for token in self.token_list: 
                token.data[:len(emb_label)] = emb_label

    def inner_structure_update(self):
        return self.token_view()

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list): 
            
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  

            inner_adj = torch.where(token_sim < self.inner_prune, torch.tensor(0.0).to(token_dot.device), token_sim)
            
            edge_index = inner_adj.nonzero().t().contiguous()

            pg_list.append(Data(x=tokens, edge_index=edge_index, y=torch.tensor([i]).long()))
            
            
            


        pg_batch = Batch.from_data_list(pg_list)
        
        return pg_batch

    def token_view(self, ):
        """
        each token group is viewed as a prompt sub-graph.
        turn the all groups of tokens as a batch of prompt graphs.
        :return:
        """
        pg_list = []
        for i, tokens in enumerate(self.token_list): 
            
            token_dot = torch.mm(tokens, torch.transpose(tokens, 0, 1))
            token_sim = torch.sigmoid(token_dot)  

            inner_adj = torch.where(token_sim < self.inner_prune, torch.tensor(0.0).to(token_dot.device), token_sim)
            edge_index = inner_adj.nonzero().t().contiguous()

            
            g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=tokens.shape[0])
            g.ndata['feat'] = tokens
            pg_list.append(g)


        
        pg_batch = dgl.batch(pg_list)
        return pg_batch


class HeavyPrompt(LightPrompt):
    def __init__(self, token_dim, token_num, group_num, cross_prune=0.1, inner_prune=0.01, emb_label=None, cat_emb=False, text_prompt=False):
        
        super(HeavyPrompt, self).__init__(token_dim, token_num_per_group=token_num, group_num=group_num, inner_prune=0.1, emb_label=emb_label, 
                                          cat_emb=cat_emb, text_prompt=text_prompt)  
        self.cross_prune = cross_prune

    def convert_dgl_to_pyg(self, dgl_graph):
        edge_index = dgl_graph.edges()
        edge_index = torch.stack([edge_index[1], edge_index[0]])

        if 'feat' in dgl_graph.ndata:
            x = dgl_graph.ndata['feat']
        else:
            x = None

        if 'feat' in dgl_graph.edata:
            edge_attr = dgl_graph.edata['feat']
        else:
            edge_attr = None

        pyg_graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return pyg_graph

    def forward_raw(self, graph_batch: Batch):
        """
        TODO: although it recieves graph batch, currently we only implement one-by-one computing instead of batch computing
        TODO: we will implement batch computing once we figure out the memory sharing mechanism within PyG
        :param graph_batch:
        :return:
        """
        
        

        pg = self.inner_structure_update()  

        if isinstance(graph_batch, dgl.DGLGraph):
            graph_pyg_list = []
            for dgl_graph in dgl.unbatch(graph_batch):
                graph_pyg = self.convert_dgl_to_pyg(dgl_graph)
                graph_pyg_list.append(graph_pyg.to(pg.x.device))
            graph_batch = Batch.from_data_list(graph_pyg_list)

        inner_edge_index = pg.edge_index
        token_num = pg.x.shape[0]

        re_graph_list = []
        for g in Batch.to_data_list(graph_batch):
            g_edge_index = g.edge_index + token_num 
            
            
            
            cross_dot = torch.mm(pg.x, torch.transpose(g.x, 0, 1))
            cross_sim = torch.sigmoid(cross_dot)  
            cross_adj = torch.where(cross_sim < self.cross_prune, torch.tensor(0.0).to(pg.x.device), cross_sim)
            
            cross_edge_index = cross_adj.nonzero().t().contiguous()
            cross_edge_index[1] = cross_edge_index[1] + token_num
            
            x = torch.cat([pg.x, g.x], dim=0)
            y = g.y

            edge_index = torch.cat([inner_edge_index, g_edge_index, cross_edge_index], dim=1)
            data = Data(x=x, edge_index=edge_index, y=y)
            re_graph_list.append(data)

        graphp_batch = Batch.from_data_list(re_graph_list)
        return graphp_batch

    def forward(self, graph_batch: Batch):
        pg = self.inner_structure_update()

        inner_edge_index = pg.edges()
        token_num = pg.num_nodes()

        re_graph_list = []
        for g in dgl.unbatch(graph_batch):
            g_edge_index = g.edges()
            g_edge_index = (g_edge_index[0] + token_num, g_edge_index[1] + token_num)  

            cross_dot = torch.mm(pg.ndata['feat'], torch.transpose(g.ndata['feat'], 0, 1))
            cross_sim = torch.sigmoid(cross_dot)
            cross_adj = torch.where(cross_sim < self.cross_prune, torch.tensor(0.0).to(pg.device), cross_sim)

            
            src, dst = torch.nonzero(cross_adj, as_tuple=True)
            cross_edge_index = (src, dst + token_num)

            
            x = torch.cat([pg.ndata['feat'], g.ndata['feat']], dim=0)
            

            
            edge_index = (torch.cat([inner_edge_index[0], g_edge_index[0], cross_edge_index[0]], dim=0),
                        torch.cat([inner_edge_index[1], g_edge_index[1], cross_edge_index[1]], dim=0))

            
            data = dgl.graph((edge_index[0], edge_index[1]))
            data.ndata['feat'] = x
            re_graph_list.append(data)

        return dgl.batch(re_graph_list)

class FrontAndHead(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=16, num_classes=2,
                 task_type="multi_label_classification",
                 token_num=10, cross_prune=0.1, inner_prune=0.3):

        super().__init__()

        self.PG = HeavyPrompt(token_dim=input_dim, token_num=token_num, group_num=1, cross_prune=cross_prune,
                              inner_prune=inner_prune)

        if task_type == 'multi_label_classification':
            self.answering = torch.nn.Sequential(
                torch.nn.Linear(hid_dim, num_classes),
                torch.nn.Softmax(dim=1))
        elif task_type == 'multi_class_classification_mean':
            self.answering = MeanAnwser(hid_dim, num_classes)
        else:
            raise NotImplementedError
    
    def to(self, device="cpu"):
        self.PG = self.PG.to(device)
        self.answering = self.answering.to(device)

    def forward(self, graph_batch, gnn):
        prompted_graph = self.PG(graph_batch)
        
        graph_emb = gnn.inference_w_grad(prompted_graph, gnn.device, 256)
        pre = self.answering(prompted_graph, x=graph_emb)

        return pre

if __name__ == '__main__':
    pass
