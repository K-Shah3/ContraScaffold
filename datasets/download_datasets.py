from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import os

def download_dataset(name, root='datasets/'):
    dataset = PygGraphPropPredDataset(name = name, root = root)
    

if __name__ == "__main__":
    dataset_stem = 'ogbg-mol'
    dataset_names = ['tox21', 'toxcast', 'muv', 
                    'bace', 'bbbp', 'clintox', 
                    'sider', 'esol', 'freesolv', 'lipo']
    
    for dataset_name in dataset_names:
        name = dataset_stem + dataset_name
        dataset = PygGraphPropPredDataset(name = name, root='')
    