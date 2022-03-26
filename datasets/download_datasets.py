from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import os

OGBG_DATASET_STEM = 'ogbg-mol'


def download_ogbg_datasets(root=''):
    dataset_names = ['tox21', 'toxcast', 'muv', 
                    'bace', 'bbbp', 'clintox', 
                    'sider', 'esol', 'freesolv', 'lipo']
    for dataset_name in dataset_names:
        name = OGBG_DATASET_STEM + dataset_name
        PygGraphPropPredDataset(name = name, root= root)

def get_ogbg_dataset(dataset_name, root = ''):
    name = OGBG_DATASET_STEM + dataset_name
    return PygGraphPropPredDataset(name = name, root= root)

def get_ogbg_dataset_loaders(dataset_name, batch_size=32, root='', shuffle_train=True, shuffle_valid=False, shuffle_test=False):
    dataset = get_ogbg_dataset(dataset_name, root)
    split_idx = dataset.get_idx_split() 
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=shuffle_train)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=shuffle_valid)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=shuffle_test)
    return train_loader, valid_loader, test_loader
    

    

if __name__ == "__main__":
    # run this file to download ogbg datasets locally under the datasets/ directory
    # when running from this directory use root = ''
    download_ogbg_datasets()
    