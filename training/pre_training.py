from sklearn import config_context
import datasets.fine_tuning.ogb.download_datasets as download_ogb
import yaml
from torch_geometric.loader import DataLoader
from method.gnn_models import GNNGraphPred, GNN, GNNGraphCL
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr
import os
import datetime
from utils import get_device
# from dig.sslgraph.evaluation import NodeUnsupervised
import torch_geometric
from method.evaluator import NodeUnsupervised


def main():
    # set up
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    device = get_device(config['gpu'])
    torch.manual_seed(config['runseed'])
    np.random.seed(config['runseed'])
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # device = 'cpu'
    
    # set up dataset
    dataset_config = config['pre_train_dataset']
    dataset_name = dataset_config['ogbg_dataset_name']
    print(f'dataset name: {dataset_name}')
    dataset_path = dataset_config['ogbg_data_path']
    num_tasks, task_type, metric = download_ogb.get_num_type_metric_of_ogbg_dataset(dataset_name)
    dataset = download_ogb.get_ogbg_dataset(dataset_name=dataset_name, root=dataset_path)
    train_loader, valid_loader, test_loader = download_ogb.get_ogbg_dataset_loaders(dataset_name, batch_size=dataset_config['batch_size'], root=dataset_path)
    train_mask, valid_mask, test_mask = download_ogb.get_ogbg_dataset_masks(dataset_name, root=dataset_path)

    # set up load and save directories
    load_save_config = config["load_save_pre_train"]
    if load_save_config["load_model_dir"]:
        load_dir = 'results/' + load_save_config["load_model_dir"]
        load_file = load_dir + load_save_config["load_model_name"]
    else:
        load_file = None
    
    save_dir = 'results/pre_training/' + dataset_name + "/"
    if not os.path.exists(save_dir):
        os.system(f'mkdir -p {save_dir}')

    # set up encoder
    encoder_config = config['pre_train_encoder']
    encoder = GNN(encoder_config['num_layer'], emb_dim=encoder_config['emb_dim'], JK=encoder_config["JK"], 
                    drop_ratio=encoder_config["dropout_ratio"], gnn_type=encoder_config["gnn_type"])
    if not encoder_config['aug1']:
        aug1 = None
    else:
        aug1 = encoder_config['aug1']

    if not encoder_config['aug2']:
        aug2 = None
    else:
        aug2 = encoder_config['aug2']
    graphcl = GNNGraphCL(encoder_config['emb_dim'], aug_1=aug1, aug_2=aug2, tau=encoder_config['tau'])
    evaluator = NodeUnsupervised(dataset, train_mask=train_mask, val_mask=valid_mask, test_mask=test_mask, log_interval=encoder_config['log_interval'])
    evaluator.evaluate(learning_model=graphcl, encoder=encoder)

def test():
    # set up
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    device = get_device(config['gpu'])
    torch.manual_seed(config['runseed'])
    np.random.seed(config['runseed'])
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # device = 'cpu'
    
    # set up dataset
    dataset_config = config['pre_train_dataset']
    dataset_name = dataset_config['ogbg_dataset_name']
    print(f'dataset name: {dataset_name}')
    dataset_path = dataset_config['ogbg_data_path']
    num_tasks, task_type, metric = download_ogb.get_num_type_metric_of_ogbg_dataset(dataset_name)
    dataset = download_ogb.get_ogbg_dataset(dataset_name=dataset_name, root=dataset_path)
    full_loader = torch_geometric.data.DataLoader(dataset, 1)
    return 

if __name__ == "__main__":
    main()
    # test()
    
    