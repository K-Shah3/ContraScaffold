import random
import torch
import numpy as np
from torch_geometric.data import Batch, Data
from abc import ABCMeta, abstractmethod

from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit import RDLogger                                                                                                                                                               
RDLogger.DisableLog('rdApp.*') 

class ViewFunction(metaclass=ABCMeta):
    
    def __call__(self, data):
        return self.views_fn(data)

    def views_fn(self, data):
        r"""Method to be called when :class:`NodeAttrMask` object is called.
        
        Args:
            data (:class:`torch_geometric.data.Data`): The input graph or batched graphs.
            
        :rtype: :class:`torch_geometric.data.Data`.  
        """
        if isinstance(data, Batch):
            dlist = [self.do_trans(d) for d in data.to_data_list()]
            return Batch.from_data_list(dlist)
        elif isinstance(data, Data):
            return self.do_trans(data)
    
    @abstractmethod
    def do_trans(self, data):
        pass

class IdentityViewFunction(ViewFunction):
    '''Identity view function on the given graph or batched graphs
    Class objects callable via method :meth:`views_fn`.
    '''
    def do_trans(self, data):
        return data
    
class NodeAttrMask(ViewFunction):
    '''Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    '''
    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, return_mask=False, device='cpu'):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.return_mask = return_mask
        self.device = device
    
    def do_trans(self, data):
        
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()

        if self.mode == 'whole':
            mask = torch.zeros(node_num).to(self.device)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            rand = np.random.normal(loc=self.mask_mean, scale=self.mask_std, 
                                                        size=(mask_num, feat_dim))
            # TODO: check if this the right way to be masking
            rand[:, 0][rand[:, 0] <= -1] = 0
            rand[:, 1][rand[:, 1] <= -1] = 0
            rand_torch = torch.tensor(rand, dtype=torch.long).to(self.device)
            
            x[idx_mask] = rand_torch

            # x1_unique = torch.unique(x[:, 0])
            # x2_unique = torch.unique(x[:, 1])
            # print(f'unique x1: {x1_unique}')
            # print(f'unique x2: {x2_unique}')
            # x[idx_mask] = torch.tensor(np.random.normal(loc=self.mask_mean, scale=self.mask_std, 
            #                                             size=(mask_num, feat_dim)), dtype=torch.long)
            mask[idx_mask] = 1

        elif self.mode == 'partial':
            mask = torch.zeros((node_num, feat_dim))
            for i in range(node_num):
                for j in range(feat_dim):
                    if random.random() < self.mask_ratio:
                        x[i][j] = torch.tensor(np.random.normal(loc=self.mask_mean, 
                                                                scale=self.mask_std), dtype=torch.float32)
                        mask[i][j] = 1

        elif self.mode == 'onehot':
            mask = torch.zeros(node_num)
            mask_num = int(node_num * self.mask_ratio)
            idx_mask = np.random.choice(node_num, mask_num, replace=False)
            x[idx_mask] = torch.tensor(np.eye(feat_dim)[np.random.randint(0, feat_dim, size=(mask_num))], dtype=torch.float32)
            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(self.mode))

        if self.return_mask:
            return Data(x=x, edge_index=data.edge_index, mask=mask, edge_attr=data.edge_attr)
        else:
            return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr)


class ScaffoldAwareNodeAttrMask(ViewFunction):
    '''Node attribute masking on the given graph or batched graphs. 
    Class objects callable via method :meth:`views_fn`.
    
    Args:
        mode (string, optinal): Masking mode with three options:
            :obj:`"whole"`: mask all feature dimensions of the selected node with a Gaussian distribution;
            :obj:`"partial"`: mask only selected feature dimensions with a Gaussian distribution;
            :obj:`"onehot"`: mask all feature dimensions of the selected node with a one-hot vector.
            (default: :obj:`"whole"`)
        mask_ratio (float, optinal): The ratio of node attributes to be masked. (default: :obj:`0.1`)
        mask_mean (float, optional): Mean of the Gaussian distribution to generate masking values.
            (default: :obj:`0.5`)
        mask_std (float, optional): Standard deviation of the distribution to generate masking values. 
            Must be non-negative. (default: :obj:`0.5`)
    '''
    def __init__(self, mode='whole', mask_ratio=0.1, mask_mean=0.5, mask_std=0.5, return_mask=False, device='cpu'):
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.mask_mean = mask_mean
        self.mask_std = mask_std
        self.return_mask = return_mask
        self.device = device

    def get_non_scaffold_indices(self, smiles):
        """Given the smiles of the molecule, calculate which indices of the data are not part of the scaffold, and 
        therefore can be masked for the augmentations"""
        smiles_of_scaffold = MurckoScaffoldSmiles(smiles)
        mol = Chem.MolFromSmiles(smiles)
        mol_with_h = Chem.AddHs(mol)
        scaffold = Chem.MolFromSmiles(smiles_of_scaffold)
        scaffold_indices = mol.GetSubstructMatch(scaffold)
        non_scaffold_indices = [i for i in range(mol.GetNumAtoms()) if i not in scaffold_indices]
        num_atoms_mol = mol.GetNumAtoms()
        num_atoms_mol_h = mol_with_h.GetNumAtoms()
        extra_atoms = num_atoms_mol_h - num_atoms_mol
        non_scaffold_indices += [i + num_atoms_mol for i in range(extra_atoms)]

        return non_scaffold_indices

    def do_trans(self, data):
        
        node_num, feat_dim = data.x.size()
        x = data.x.detach().clone()
        if not data.smiles:
            raise Exception("Data does not have corresponding smiles")
        smiles = data.smiles
        non_scaffold_indices = self.get_non_scaffold_indices(smiles)
        non_scaffold_node_num = len(non_scaffold_indices)
        
        if self.mode == 'whole':
            mask = torch.zeros(node_num).to(self.device)
            mask_num = int(non_scaffold_node_num * self.mask_ratio)
            idx_mask = np.random.choice(non_scaffold_indices, mask_num, replace=False)
            rand = np.random.normal(loc=self.mask_mean, scale=self.mask_std, 
                                                        size=(mask_num, feat_dim))
            # TODO: check if this the right way to be masking
            rand[:, 0][rand[:, 0] <= -1] = 0
            rand[:, 1][rand[:, 1] <= -1] = 0
            rand_torch = torch.tensor(rand, dtype=torch.long).to(self.device)
            
            x[idx_mask] = rand_torch
            mask[idx_mask] = 1

        else:
            raise Exception("Masking mode option '{0:s}' is not available!".format(self.mode))

        if self.return_mask:
            return Data(x=x, edge_index=data.edge_index, mask=mask, edge_attr=data.edge_attr, smiles=smiles)
        else:
            return Data(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, smiles=smiles)



