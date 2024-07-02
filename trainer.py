import os
import torch
import logging
import copy
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoModel, get_scheduler
from tqdm.auto import tqdm
from tqdm import trange
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import wandb
import dgl
import warnings

from tqdm import tqdm, trange

from data import *
from model import BaseLM, JointModel
from utils.functions import create_optimizer, get_current_lr, set_random_seed, drop_edge, l2_normalize
from utils.evaluation import node_classification_evaluation, link_prediction_evaluation

from torch_geometric.data import Batch
from sklearn.exceptions import ConvergenceWarning

from graph_prompt_trainer import model_create, model_components, meta_train_maml

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Trainer():
    def __init__(self, args) -> None:
        self.lm_type = args.lm_type
        
        self.batch_size = args.batch_size
        self.num_epochs = args.num_epochs
        self.device = args.device if args.device >= 0 else "cpu"
        self.lr = args.lr
        self.task = args.task
        self.eval_only = args.eval_only
        self.args = args

        self.g_val_acc_lm = self.g_test_acc_lm = 0
        self.g_val_acc_gnn = self.g_test_acc_gnn = 0
        self.fw_val_acc = self.fw_test_acc = 0

    def evaluate_LR(self, emb, mode, N, K, model):
        
        dataset = IterFewShotDataset(self.tag_data, mode, self.args.num_repeat, self.args.eval_num_tasks, N, K, self.args.k_qry, need_val=True, woprompt=True)
        eval_loader = DataLoader(dataset, batch_size=None)
        
        accs_repeat = []
        with torch.no_grad(): 
            for rep in tqdm(range(self.args.num_repeat)):
                accs = []
                for batch, batch_item in eval_loader:
                    batch_nodes, sqt_idx, val_idx, qrt_idx = batch_item
                    
                    
                    clf = LogisticRegression(penalty='l2',
                                                    random_state=0,
                                                    C=1.0,
                                                    solver='lbfgs',
                                                    max_iter=100,
                                                    multi_class='multinomial')
                    spt_emb = emb[batch_nodes][sqt_idx]
                    qry_emb = emb[batch_nodes][qrt_idx]
                    # spt_emb = l2_normalize(spt_emb).cpu().numpy()
                    # qry_emb = l2_normalize(qry_emb).cpu().numpy()
                    spt_emb = spt_emb.cpu().numpy()
                    qry_emb = qry_emb.cpu().numpy()
                    spt_labels = np.arange(N)
                    spt_labels = np.repeat(spt_labels, K)
                    qry_labels = np.arange(N)
                    qry_labels = np.repeat(qry_labels, self.args.k_qry)
                    clf.fit(spt_emb, spt_labels)
                    preds = clf.predict(qry_emb)

                    acc = accuracy_score(qry_labels, preds)
                    accs.append(acc)
                accs = np.array(accs)
                acc_mean = np.mean(accs)
                accs_repeat.append(acc_mean)
                dataset.current_round += 1
        accs_repeat = np.array(accs_repeat)
        acc_mean = np.mean(accs_repeat)
        acc_std = np.std(accs_repeat)
        return acc_mean, acc_std, emb
    
    def evaluate_fw(self, embedding, emb_type, model, name=""):
        test_accs = []
        # for ii in range(len(self.args.n_way)):
            # N_way = self.args.n_way[ii]
            # K_sqt = self.args.k_sqt[ii]
        for ii in range(1):
            N_way, K_sqt = [int(_) for _ in self.args.few_shot_setting.split(",")]
            test_acc, test_acc_std, _ = self.evaluate_LR(embedding, "test", N_way, K_sqt, model)
            if emb_type == "LM_emb":
                print(f"The P2TAG(LM) few shot test_acc {N_way}-way {K_sqt}-shot: {test_acc*100:.2f}±{test_acc_std*100:.2f}")
            elif emb_type == "GNN_emb":
                print(f"The P2TAG(GNN) few shot test_acc {N_way}-way {K_sqt}-shot: {test_acc*100:.2f}±{test_acc_std*100:.2f}")
            test_accs.append(test_acc)
        if self.args.logging:
            mean_test_accs = np.array(test_accs).mean()
            if emb_type == "LM_emb":
                wandb.log({
                            "fw_test_acc_lm": mean_test_accs,
                        })
            elif emb_type == "GNN_emb":
                wandb.log({
                            "fw_test_acc_gnn": mean_test_accs,
                        })
        return embedding
    
    def w_p_meta_train(self, emb, pre_trained_gnn, emb_type, name=""):
        pass

    def generate_param_path(self, dir_name, args):
        """
        Generate a unique file path for saving/loading parameters based on the provided arguments.
        """
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        param_path = dir_name + "/"
        param_path += f"{args.dataset_name}_few_shot_setting_{args.few_shot_setting}_"
        param_path += f"num_tasks_{args.num_tasks}_token_num_{args.token_num}_"
        param_path += f"adapt_steps_{args.adapt_steps}_meta_epochs_{args.meta_epochs}"
        param_path += ".pt"
        return param_path

    def save_parameters(self, maml, path):
        """
        Save the parameters of the MAML model.
        """
        torch.save({
            'PG_weight': maml.module.PG.state_dict(),
            'answering_weight': maml.module.answering.state_dict()
        }, path)
        return maml.module.PG.state_dict(), maml.module.answering.state_dict() 
    
    def load_paramters(self, path):
        checkpoint = torch.load(path)
        return checkpoint['PG_weight'], checkpoint['answering_weight']

    def evaluate_fw_w_p(self, emb, pre_trained_gnn, emb_type="", name=""):
        self.epoch_tracker = {
            'val_acc_epochs': [],
            'max_acc_epochs': [],
            'last_acc_epochs': []
        }

        if self.args.logging:
            wandb.init(project="TAG-Exp", entity=self.args.run_entity, config=self.args.__dict__, settings=wandb.Settings(start_method='fork'))
        else:
            wandb.init(project="TAG-Exp", entity=self.args.run_entity, config=self.args.__dict__, mode="disabled")
        N, K = [int(_) for _ in self.args.few_shot_setting.split(",")]
        print(f"The few shot setting is {N}-way {K}-shot")
        print(f"args.token_num: {self.args.token_num}")

        
        PG_weight, answering_weight = None, None
        if self.args.meta_train:
            parameter_path = self.generate_param_path(dir_name="meta_output", args=self.args)
            print(f"【Meta-train】Start meta training, saving path:{parameter_path}")
            if os.path.exists(parameter_path):
                PG_weight, answering_weight = self.load_paramters(parameter_path)
                print("【Meta-train】Parameters loaded successfully. Skip meta training")
            else:
                dataset = IterFewShotDataset(self.tag_data, "train", self.args.num_repeat, self.args.num_tasks, N, K, self.args.k_qry, 
                                            emb=emb, is_graph=True)
                train_loader = DataLoader(dataset, batch_size=None)

                maml, opt, lossfn = model_components(self.args, N, K)
                
                maml.module.to(pre_trained_gnn.device)
                meta_train_maml(self.args.meta_epochs, maml, pre_trained_gnn, lossfn, opt, train_loader, self.args.adapt_steps, N, K, self.args)
                PG_weight, answering_weight = self.save_parameters(maml, parameter_path)
                print(f"【Meta-train】Parameters saved.")
        else:
            print(f"【Meta-train】meta-training is disabled")

        
        dataset = IterFewShotDataset(self.tag_data, "test", self.args.num_repeat, self.args.eval_num_tasks, N, K, self.args.k_qry, 
                                     emb=emb, is_graph=True, max_sample_node=self.args.max_sample_node, model=pre_trained_gnn if self.args.label_as_init else None)
        
        
        eval_loader = DataLoader(dataset, batch_size=None)

        accs_repeat = []
        max_accs_repeat = []
        last_accs_repeat = []
        for rep in tqdm(range(self.args.num_repeat)):
            accs = []
            max_accs = []
            last_accs = []
            acc_epochs, max_acc_epochs, last_acc_epochs = [], [], []
            for batch_id, (batch, batch_item) in tqdm(enumerate(eval_loader), total=len(eval_loader)):
                
                batch_nodes, sqt_idx, val_idx, qrt_idx, sqt_graphs, val_graphs, qrt_graphs, emb_label = batch_item
                sqt_emb, qrt_emb, val_emb = l2_normalize(emb[batch_nodes][sqt_idx]), l2_normalize(emb[batch_nodes][qrt_idx]), l2_normalize(emb[batch_nodes][val_idx])
                emb_label = l2_normalize(emb_label) if self.args.label_as_init else None
                acc, acc_epoch, max_acc, max_acc_epoch, last_acc, last_acc_epoch = self.w_p_test(self.args, pre_trained_gnn, sqt_emb, qrt_emb, val_emb, emb_label, sqt_graphs, qrt_graphs, val_graphs, self.device, N, K, logging=False, PG_weight=PG_weight, answering_weight=answering_weight)
                accs.append(acc)
                max_accs.append(max_acc)
                last_accs.append(last_acc)
                acc_epochs.append(acc_epoch)
                max_acc_epochs.append(max_acc_epoch)
                last_acc_epochs.append(last_acc_epoch)
                wandb.log({f"Repeat_{rep}/batch_acc": acc,
                        f"Repeat_{rep}/batch_val_acc": max_acc,
                        f"Repeat_{rep}/batch_last_acc": last_acc,
                        f"Repeat_{rep}/batch_acc_epoch": acc_epoch,
                        f"Repeat_{rep}/batch_val_acc_epoch": max_acc_epoch,
                        f"Repeat_{rep}/batch_last_acc_epoch": last_acc_epoch})
                
            
            batch_acc, batch_acc_epoch, batch_max_acc, batch_max_acc_epoch, batch_last_acc, batch_last_acc_epoch = np.array(accs), np.array(acc_epochs), np.array(max_accs), np.array(max_acc_epochs), np.array(last_accs), np.array(last_acc_epochs)
            accs_repeat.append(batch_acc.mean())
            max_accs_repeat.append(batch_max_acc.mean())
            last_accs_repeat.append(batch_last_acc.mean())
            wandb.log({f"Repeat_{rep}/acc_mean": batch_acc.mean(),
                        f"Repeat_{rep}/acc_std": batch_acc.std(),
                        f"Repeat_{rep}/max_acc_mean": batch_max_acc.mean(),
                        f"Repeat_{rep}/max_acc_std": batch_max_acc.std(),
                        f"Repeat_{rep}/last_acc_mean": batch_last_acc.mean(),
                        f"Repeat_{rep}/last_acc_std": batch_last_acc.std(),
                        f"Repeat_{rep}/acc_epoch": batch_acc_epoch.mean(),
                        f"Repeat_{rep}/max_acc_epoch": batch_max_acc_epoch.mean(),
                        f"Repeat_{rep}/last_acc_epoch": batch_last_acc_epoch.mean()})
            dataset.current_round += 1
            print(f"[Fold {rep}/{self.args.num_repeat}]: {batch_acc.mean()*100:.2f} ± {batch_acc.std()*100:.2f}({batch_acc_epoch.mean()} ± {batch_acc_epoch.std()*100:.2f}) {batch_max_acc.mean()*100:.2f} ± {batch_max_acc.std()*100:.2f}({batch_max_acc_epoch.mean()} ± {batch_max_acc_epoch.std()*100:.2f}) {batch_last_acc.mean()*100:.2f} ± {batch_last_acc.std()*100:.2f}({batch_last_acc_epoch.mean()} ± {batch_last_acc_epoch.std()*100:.2f}")
        accs_repeat = np.array(accs_repeat)
        acc_mean = np.mean(accs_repeat)
        acc_std = np.std(accs_repeat)

        max_accs_repeat = np.array(max_accs_repeat)
        max_accs_mean, max_accs_std = max_accs_repeat.mean(), max_accs_repeat.std()

        last_accs_repeat = np.array(last_accs_repeat)
        last_accs_mean, last_accs_std = last_accs_repeat.mean(), last_accs_repeat.std()
        wandb.log({f"{self.args.num_repeat}_fold_acc_mean": acc_mean,
                    f"{self.args.num_repeat}_fold_acc_std": acc_std,
                    f"{self.args.num_repeat}_fold_max_acc_std": max_accs_std,
                    f"{self.args.num_repeat}_fold_max_acc_mean": max_accs_mean,
                    f"{self.args.num_repeat}_fold_last_acc_mean": last_accs_mean,
                    f"{self.args.num_repeat}_fold_last_acc_std": last_accs_std})
        wandb.log({f"{self.args.num_repeat}_fold_acc_std": f"{acc_mean*100:.2f}±{acc_std*100:.2f}",
                    f"{self.args.num_repeat}_fold_max_acc_std": f"{max_accs_mean*100:.2f}±{max_accs_std*100:.2f}",
                    f"{self.args.num_repeat}_fold_last_acc_std": f"{last_accs_mean*100:.2f}±{last_accs_std*100:.2f}"})
        # print(f"few shot test_acc {N}-way {K}-shot: {acc_mean*100:.2f}±{acc_std*100:.2f} {max_accs_mean*100:.2f}±{max_accs_std*100:.2f} {last_accs_mean*100:.2f}±{last_accs_std*100:.2f}")
        print(f"Prompting few shot test_acc {N}-way {K}-shot: {max_accs_mean*100:.2f}±{max_accs_std*100:.2f}")
        print()
        return emb

    def w_p_test(self, args, gnn, sqt_emb, qrt_emb, val_emb, emb_label, sqt_graphs, qrt_graphs, val_graphs, device, N_way, K_shot, logging=True, PG_weight=None, answering_weight=None):
        
        
        spt_labels = torch.repeat_interleave(torch.arange(N_way), K_shot).to(gnn.device)
        qry_labels = torch.repeat_interleave(torch.arange(N_way), self.args.k_qry).to(gnn.device)
        val_labels = torch.repeat_interleave(torch.arange(N_way), self.args.k_qry).to(gnn.device)

        PG, answering, opi_answer, opi_pg, lossfn = model_create(N_way, tune_answer=True, args=self.args, emb_label=emb_label)
        
        if PG_weight is not None and answering_weight is not None:
            PG.load_state_dict(PG_weight)
            answering.load_state_dict(answering_weight)

        

        PG = PG.to(f"cuda:{device}")
        answering = answering.to(f"cuda:{device}")

        train_loader, test_loader = dgl.batch(sqt_graphs), dgl.batch(qrt_graphs)
        val_loader = dgl.batch(val_graphs)

        outer_epoch = self.args.prompt_epochs
        answer_epoch = 1 
        prompt_epoch = 1 

        min_val_loss = 1e6
        max_val_acc = 0

        max_acc, eval_loss, last_acc = 0, 0, 0
        max_acc_epoch, val_acc_epoch, last_acc_epoch = 0, 0, 0
        for i in range(1, outer_epoch+1): 
            
            if logging:
                print(("{}/{} frozen gnn | frozen prompt | *tune answering function...".format(i, outer_epoch)))
            answering.train()
            PG.eval()
            self.train_one_outer_epoch(answer_epoch, train_loader, spt_labels, sqt_emb, emb_label, opi_answer, lossfn, gnn, PG, answering)

            
            if logging:
                print("{}/{}  frozen gnn | *tune prompt |frozen answering function...".format(i, outer_epoch))
            answering.eval()
            PG.train()
            train_loss = self.train_one_outer_epoch(prompt_epoch, train_loader, spt_labels, sqt_emb, emb_label, opi_pg, lossfn, gnn, PG, answering)

            answering.eval()
            PG.eval()
            val_loss, val_acc = self.train_one_outer_epoch(1, val_loader, val_labels, val_emb, emb_label, opi_pg, lossfn, gnn, PG, answering, val=True)
            test_loss, test_acc = self.train_one_outer_epoch(1, test_loader, qry_labels, qrt_emb, emb_label, opi_pg, lossfn, gnn, PG, answering, val=True)
            wandb.log({"epoch": i, "train_loss": train_loss, "val_loss": val_loss, "test_acc": val_acc})
            if max_acc < test_acc:
                max_acc = test_acc
                max_acc_epoch = i
                
                
            
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                
                
                val_acc_epoch = i
                acc = test_acc

            last_acc = test_acc
            last_acc_epoch = i

        wandb.log({"test_acc": acc})
        return acc, val_acc_epoch, max_acc, max_acc_epoch, last_acc, last_acc_epoch 
    
    def train_one_outer_epoch(self, epoch, train_loader, train_label, train_emb, emb_label, opi, lossfn, gnn, PG, answering, val=False):
        for j in range(1, epoch + 1):
            running_loss = 0.
            pre_list = []
            prompt_graph = PG(train_loader.to(f"cuda:{gnn.device}"))
            text_prompt = PG.text_prompt(train_emb) if self.args.LM_as_init else train_emb
            graph_emb = gnn.inference_w_grad(prompt_graph, gnn.device, 256)
            pre = answering(prompt_graph, graph_emb.to(gnn.device), text_prompt, emb_label=emb_label) 
                
            train_loss = lossfn(pre, train_label)
            if val:
                
                train_acc = accuracy_score(train_label.cpu().numpy(), pre.argmax(dim=1).cpu().numpy()) 
                return train_loss.item() / len(dgl.unbatch(train_loader)), train_acc
            else:
                opi.zero_grad()
                train_loss.backward()
                opi.step()
                running_loss += train_loss.item()
        return running_loss / (len(dgl.unbatch(train_loader))) * epoch

    def acc_f1_over_batches(self, test_loader, test_label, PG, gnn, answering, num_class, device='cpu'):
        pre_list = []
        prompted_graph_list = []
        for (batch_id, train_batch), _ in zip(enumerate(dgl.unbatch(test_loader)), test_label):  
            

            train_batch = train_batch.to(f"cuda:{gnn.device}")
            prompted_graph = PG(train_batch)
            prompted_graph_list.append(prompted_graph)
        graph_emb = gnn.inference(dgl.batch(prompted_graph_list), gnn.device, 256)
        
        pre = answering(dgl.batch(prompted_graph_list), graph_emb.to(gnn.device)) 
        acc = accuracy_score(test_label.cpu().numpy(), pre.argmax(dim=1).cpu().numpy())
        return acc
    
    def evaluate(self, model, name=""):
        dataset = TAGDataset(self.tag_data, self.args.lm_path)
        eval_loader = DataLoader(dataset, shuffle=False, batch_size=self.args.eval_batch_size)
        model.eval()
        
        # P2TAG(LM)
        output = []
        if not os.path.exists(f"./output/lm_output_{self.args.dataset_name}.pt"):
            os.makedirs("./output/", exist_ok=True)
            for batch, _ in tqdm(eval_loader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                emb = model.emb(batch)
                output.append(emb.cpu())
            output = torch.cat(output, 0) 
            torch.save(output, f"./output/lm_output_{self.args.dataset_name}.pt")
        else:
            output = torch.load(f"./output/lm_output_{self.args.dataset_name}.pt")
        
        self.evaluate_fw(output, "LM_emb", model)
        
        # P2TAG(GNN)
        # if self.args.emb_type == "GNN":
        if not os.path.exists(f"./output/gnn_output_{self.args.dataset_name}.pt"):
            os.makedirs("./output/", exist_ok=True)
            graph = self.tag_data.graph.to(self.device)
            
            graph.ndata["feat"] = output.to(self.device)
            output_GNN = model.inference(graph, self.device, 256) 
            print(f"the emb is now transformed to GNN emb")
            torch.save(output_GNN, f"./output/gnn_output_{self.args.dataset_name}.pt")
        else:
            output_GNN = torch.load(f"./output/gnn_output_{self.args.dataset_name}.pt")    
        
        self.evaluate_fw(output_GNN, "GNN_emb", model)
        
        if self.args.emb_type == "GNN":
            output = output_GNN

        # P2TAG
        output = self.evaluate_fw_w_p(output.to(self.device), model)
        
        return output
    
    def evaluate_wo_train(self):
        set_random_seed(self.args.seed)
        
        self.tag_data = TAG(self.args)
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.args.lm_type, cache_dir=self.args.lm_path)
        except OSError:
            tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser("~") + "/deberta-base/")
        mask_token_id = tokenizer(tokenizer.mask_token)['input_ids'][1]
        if self.args.gnn_type == "":
            model = BaseLM(self.args, mask_token_id)
        else:
            model = JointModel(self.args, mask_token_id)
        state_dict = torch.load(self.args.eval_model_path, map_location=torch.device(self.device))
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        emb = self.evaluate(model, "initial")
        return emb

    def train(self):
        if self.args.logging:
            if self.args.run_name != "":
                wandb.init(project="TAG-Exp", entity=self.args.run_entity, name=self.args.run_name)
            else:
                wandb.init(project="TAG-Exp", entity=self.args.run_entity)
        else:
            wandb.init(project="TAG-Exp", entity=self.args.run_entity, mode="disabled")
            wandb.config.update(self.args)

        set_random_seed(self.args.seed)

        
        self.tag_data = TAG(self.args)
        
        if self.args.gnn_type == "":
            self.dataset = TAGDataset(self.tag_data, self.args.lm_path)
            dataloader = DataLoader(self.dataset, shuffle=True, batch_size=self.batch_size)
            model = BaseLM(self.args, self.dataset.mask_token_id)
        else:
            self.dataset = IterTAGDataset(self.tag_data, self.args.lm_path, self.batch_size, self.args.num_roots, self.args.length)
            dataloader = DataLoader(self.dataset, batch_size=None)
            model = JointModel(self.args, self.dataset.mask_token_id)
            

        
        
        optimizer = create_optimizer(self.args.optimizer, model, self.lr, self.args.weight_decay)
        num_training_steps = self.num_epochs * len(dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        
        model.to(self.device)

        
        latent_loss = torch.tensor(0)
        progress_bar = tqdm(range(num_training_steps))
        for epoch in range(self.num_epochs):
            count = 0
            
            for batch, batch_item in dataloader:
                model.train()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                if self.args.gnn_type == "":
                    masked_input_ids = batch_item
                    masked_input_ids = masked_input_ids.to(self.device)
                    loss = model(batch, masked_input_ids)
                else:
                    batch_nodes = batch_item
                    graph = dgl.node_subgraph(self.tag_data.graph, batch_nodes)
                    graph = graph.to(self.device)
                    drop_g1 = drop_edge(graph, self.args.drop_edge_rate)
                    drop_g2 = drop_edge(graph, self.args.drop_edge_rate)
                    loss, latent_loss = model(batch, graph, epoch=epoch, drop_g1=drop_g1, drop_g2=drop_g2)
                loss.backward()
                optimizer.step()
                
                optimizer.zero_grad()
                progress_bar.update(1)

                if self.args.logging:
                    wandb.log({
                        "current_lr": get_current_lr(optimizer),
                        "pretrain_loss": loss.item(),
                        "latent_loss": latent_loss.item(),
                    })
                progress_bar.set_description(f"# Epoch {epoch}, train_loss: {loss.item():.8f}")
                count += 1
                # if count % self.args.eval_steps == 0:
                #     emb = self.evaluate(model, f"{epoch}_{count}")
                    
            print(self.args.save_model_path)
            os.makedirs(self.args.save_model_path, exist_ok=True)
            current_time = datetime.now()
            formatted_time = current_time.strftime("%Y%m%d%H%M")
            lm_name = self.args.lm_type.split("/")[1]
            model_filename = f"model_{lm_name}_{epoch}_{formatted_time}.pt"
            torch.save(model.state_dict(), os.path.join(self.args.save_model_path, model_filename))
        
        # emb = self.evaluate(model, f"final")
        