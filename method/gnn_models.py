import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
# from dig.sslgraph.method.contrastive.model.contrastive import Contrastive
# from dig.sslgraph.method.contrastive.views_fn import NodeAttrMask, EdgePerturbation, \
#     UniformSample, RWSample, RandomView
from method.dig_contrastive import Contrastive
from method.views_fn import IdentityViewFunction, NodeAttrMask, ScaffoldAwareNodeAttrMask, UniformNodeDrop, ScaffoldAwareUniformNodeDrop

num_atom_type = 120 #including the extra mask tokens=119
num_chirality_tag = 4 # original =3. including the extra mask tokens=3
num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3 # original =3, inlcuding the extra mask tokens=3

class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not. 
        
    See https://arxiv.org/abs/1810.00826
    """
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__()
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        edge_attr_dim = edge_attr.shape[1]
        self_loop_attr = torch.zeros(x.size(0), edge_attr_dim)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        # return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)
        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)    # for latest version

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))

        #add features corresponding to self-loop edges.
        # self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index[0], x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)

class GNN(torch.nn.Module):
    """
    
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # x1_unique = torch.unique(x[:, 0])
        # x2_unique = torch.unique(x[:, 1])
        
        # print(f'unique x1: {x1_unique}')
        # print(f'unique x2: {x2_unique}')
        
        x1 = self.x_embedding1(x[:, 0])
        x2 = self.x_embedding2(x[:, 1])

        x = x1 + x2

        # x = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation

    def load_model(self, load_file):
        self.load_state_dict(torch.load(load_file))

class GNNGraphPred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """
    def __init__(self, num_layer, emb_dim, num_tasks, JK = "last", drop_ratio = 0, graph_pooling = "mean", gnn_type = "gin"):
        super(GNNGraphPred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type = gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        #self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))

class GNNGraphCL(Contrastive):
    """_summary_

    Args:
        dim (int): The embedding dimension.
        aug1 (sting, optinal): Types of augmentation for the first view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug2 (sting, optinal): Types of augmentation for the second view from (:obj:`"dropN"`, 
            :obj:`"permE"`, :obj:`"subgraph"`, :obj:`"maskN"`, :obj:`"random2"`, :obj:`"random3"`, 
            :obj:`"random4"`). (default: :obj:`None`)
        aug_ratio (float, optional): The ratio of augmentations. A number between [0,1).
        **kwargs (optinal): Additional arguments of :class:`dig.sslgraph.method.Contrastive`.
    """

    def __init__(self, dim, aug_1=None, aug_2=None, aug_ratio=0.2, device='cpu', model_checkpoint_path='models', encoder_name='enc', **kwargs):
        self.device = device
        self.encoder_name = encoder_name
        views_fn = []
        # TODO: implement other augmentations
        for aug in [aug_1, aug_2]:
            if aug is None or aug == 'identity':
                views_fn.append(IdentityViewFunction())
            elif aug == 'maskN':
                views_fn.append(NodeAttrMask(mask_ratio=aug_ratio, device=self.device))
            elif aug == 'maskNScaffold':
                views_fn.append(ScaffoldAwareNodeAttrMask(mask_ratio=aug_ratio, device=self.device))
            elif aug == 'uniformDropN':
                views_fn.append(UniformNodeDrop(drop_ratio=aug_ratio, device=self.device))
            elif aug == 'uniformDropNScaffold':
                views_fn.append(ScaffoldAwareUniformNodeDrop(drop_ratio=aug_ratio, device=self.device))
            else:
                raise Exception("Aug must be from ['maskN', 'identity', 'maskNScaffold'] or None.")
        
        super(GNNGraphCL, self).__init__(objective='NCE', 
                                        views_fn=views_fn,
                                        z_n_dim=dim,
                                        z_dim=dim,
                                        proj_out_dim=1,
                                        proj='linear',
                                        node_level=False,
                                        graph_level=True,
                                        device=self.device,
                                        model_path=model_checkpoint_path,
                                        encoder_name=self.encoder_name,
                                        **kwargs)

    def train(self, encoders, data_loader, optimizer, epochs, per_epoch_out=False):
        # GraphCL removes projection heads after pre-training
        for enc, _ in super(GNNGraphCL, self).train(encoders, data_loader, 
                                                    optimizer, epochs, per_epoch_out):
            yield enc
            