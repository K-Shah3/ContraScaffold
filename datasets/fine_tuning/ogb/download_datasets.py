from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader
import os
from torch_geometric.utils import add_self_loops

OGBG_DATASET_STEM = 'ogbg-mol'

def get_num_type_metric_of_ogbg_dataset(dataset_name):
    '''
    Returns the number of tasks, the type of task and the metric for the ogbg dataset
    See https://arxiv.org/abs/2005.00687

        Parameters:
                dataset_name (str): name of ogbg dataset (without the ogbg_mol part)

        Returns:
                task_num (int): #Tasks for the dataset
                task_type (str): 'clf' for binary classification 'reg' for regression
                metric (str): 'roc-auc' | 'ap' | 'rmse'
    '''
    task_num_dict = {'tox21':12, 'toxcast':617, 'muv':17, 
                    'bace':1, 'bbbp':1, 'clintox':2, 
                    'sider':27, 'esol':1, 'freesolv':1, 'lipo':1}

    task_type_dict = {'tox21':'clf', 'toxcast':'clf', 'muv':'clf', 
                    'bace':'clf', 'bbbp':'clf', 'clintox':'clf', 
                    'sider':'clf', 'esol':'reg', 'freesolv':'reg', 'lipo':'reg'}
    
    metric_dict = {'tox21':'roc-auc', 'toxcast':'roc-auc', 'muv':'ap', 
                    'bace':'roc-auc', 'bbbp':'roc-auc', 'clintox':'roc-auc', 
                    'sider':'roc-auc', 'esol':'rmse', 'freesolv':'rmse', 'lipo':'rmse'}

    return task_num_dict[dataset_name], task_type_dict[dataset_name], metric_dict[dataset_name]


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
    
    # dataset_names = ['tox21', 'toxcast', 'muv', 
    #                 'bace', 'bbbp', 'clintox', 
    #                 'sider', 'esol', 'freesolv', 'lipo']
    # dataset_names = ['tox21', 'toxcast', 'muv']

    # for dataset_name in dataset_names:
    #     root = ''
    #     dataset = get_ogbg_dataset(dataset_name, root = root)
    #     test = dataset[0]
    #     print(f'dataset {dataset_name}')
    #     print(test.x)
    #     print(test.y)
    #     print('----------------------')
