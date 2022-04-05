import copy
from sklearn.metrics import mean_squared_error, roc_auc_score
from scipy.stats import pearsonr
import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import DataLoader
from sklearn import preprocessing
from torch_geometric.nn import global_add_pool, global_mean_pool



class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


class NodeUnsupervised(object):
    r"""
    The evaluation interface for unsupervised graph representation learning evaluated with 
    linear classification. You can refer to `the benchmark code 
    <https://github.com/divelab/DIG/tree/dig/benchmarks/sslgraph>`_ 
    for examples of usage.
    
    Args:
        full_dataset (torch_geometric.data.Dataset): The graph classification dataset.
        train_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for training. Set to :obj:`None` if included in dataset.
            (default: :obj:`None`)
        val_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for validation. Set to :obj:`None` if included in dataset.
            (default: :obj:`None`)
        test_mask (Tensor, optional): Boolean tensor of shape :obj:`[n_nodes,]`, indicating 
            nodes for test. Set to :obj:`None` if included in dataset. (default: :obj:`None`)
        classifier (string, optional): Linear classifier for evaluation, :obj:`"SVC"` or 
            :obj:`"LogReg"`. (default: :obj:`"LogReg"`)
        log_interval (int, optional): Perform evaluation per k epochs. (default: :obj:`1`)
        device (int, or torch.device, optional): Device for computation. (default: :obj:`None`)
        **kwargs (optional): Training and evaluation configs in :meth:`setup_train_config`.
        
    Examples
    --------
    >>> node_dataset = get_node_dataset("Cora") # using default train/test split
    >>> evaluator = NodeUnsupervised(node_dataset, log_interval=10, device=0)
    >>> evaluator.evaluate(model, encoder)
    
    >>> node_dataset = SomeDataset()
    >>> # Using your own dataset or with different train/test split
    >>> train_mask, val_mask, test_mask = torch.Tensor([...]), torch.Tensor([...]), torch.Tensor([...])
    >>> evaluator = NodeUnsupervised(node_dataset, train_mask, val_mask, test_mask, log_interval=10, device=0)
    >>> evaluator.evaluate(model, encoder)
    """
    
    def __init__(self, full_dataset, train_loader, valid_loader, test_loader, clf_or_reg,
                device=None, config=None):

        self.full_dataset = full_dataset
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.clf = clf_or_reg == 'clf'
        self.device = device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, int):
            self.device = torch.device('cuda:%d'%device)
        else:
            self.device = device
        
        self.setup_train_config(config)

        # Use default config if not further specified

    def setup_train_config(self, config):

        self.p_optim = config['p_optim'] if config['p_optim'] else 'Adam'
        self.p_lr = config['p_lr'] if config['p_lr'] else 0.01
        self.p_weight_decay = config['p_weight_decay'] if config['p_weight_decay'] else 0
        self.p_epoch = config['p_epoch'] if config['p_epoch'] else 100
        self.log_interval = config['log_interval'] if config['log_interval'] else 10  
        self.eval_train = config['eval_train']      
    
    def evaluate(self, learning_model, encoder):
        r"""Run evaluation with given learning model and encoder(s).
        
        Args:
            learning_model: An object of a contrastive model (sslgraph.method.Contrastive)
                or a predictive model.
            encoder (torch.nn.Module): Trainable pytorch model or list of models.

        :rtype: (float, float)
        """
        
        if isinstance(encoder, list):
            params = [{'params': enc.parameters()} for enc in encoder]
        else:
            params = encoder.parameters()
        
        p_optimizer = self.get_optim(self.p_optim)(params, lr=self.p_lr, 
                                                   weight_decay=self.p_weight_decay)

        per_epoch_out = (self.log_interval<self.p_epoch)
        encs = []
        for enc in learning_model.train(encoder, self.train_loader, 
                                                    p_optimizer, self.p_epoch, per_epoch_out):
            encs.append(enc)
        
        return encs
        # for i, enc in enumerate(learning_model.train(encoder, self.train_loader, 
        #                                              p_optimizer, self.p_epoch, per_epoch_out)):
            # if not per_epoch_out or (i+1)%self.log_interval==0:
            #     print("====Evaluation")
            #     if self.clf:
            #         if self.eval_train:
            #             train_auc = self.clf_eval(enc.to(self.device), self.train_loader)
            #         else:
            #             print("omit the training accuracy computation")
            #             train_auc = 0
                    
            #         valid_auc = self.clf_eval(enc.to(self.device), self.valid_loader)
            #         test_auc = self.clf_eval(enc.to(self.device), self.test_loader)

            #         print("train auc: %f val auc: %f test auc: %f" %(train_auc, valid_auc, test_auc))
            #     else:
            #         if self.pre_training_config['eval_train']:
            #             train_mse, train_cor = self.reg_eval(enc.to(self.device), self.train_loader)
            #         else:
            #             print("omit the training accuracy computation")
            #             train_mse, train_cor = 0, 0
                    
            #         val_mse, val_cor = self.reg_eval(enc.to(self.device), self.valid_loader)
            #         test_mse, test_cor = self.reg_eval(enc.to(self.device), self.test_loader)
                    
            #         print("mse train: %f mse val: %f mse test: %f" %(train_mse, val_mse, test_mse))
            #         print("cor train: %f cor val: %f cor test: %f" %(train_cor, val_cor, test_cor))
            # pass
    
    def clf_eval(self, model, loader):
        model.eval()
        y_true = []
        y_scores = []
        
        for batch in loader:
            batch = batch.to(self.device)
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr)

            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim = 0).cpu().numpy()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()
        roc_list = []
        for i in range(y_true.shape[1]):
            #AUC is only defined when there is more than one class.
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
                is_valid = y_true[:,i]**2 >= 0
                # roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
                roc_list.append(roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i]))

        if len(roc_list) < y_true.shape[1]:
            print("Some target is missing!")
            print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

        return sum(roc_list)/len(roc_list) #y_true.shape[1]    
    
    def reg_eval(self, model, loader):
        model.eval()
        y_true = []
        y_scores = []

        for batch in loader:
            batch = batch.to(self.device)
            
            with torch.no_grad():
                pred = model(batch.x, batch.edge_index, batch.edge_attr)

            y_true.append(batch.y.view(pred.shape))
            y_scores.append(pred)

        y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
        y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
        mse = mean_squared_error(y_true, y_scores)
        cor = pearsonr(y_true, y_scores)[0]
        return mse, cor


        
    
    def get_optim(self, optim):
        
        optims = {'Adam': torch.optim.Adam}
        
        return optims[optim]

