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
    
    if load_file:
        encoder.load_model(load_file + '.pth')
        encoder_status = 'loaded'
    else:
        encoder_status = 'scratch'
    
    encoder_name = dataset_name + '_do_' + str(encoder_config['dropout_ratio']) + '_seed_' + str(config['runseed']) + '_JK_' + str(encoder_config['JK'])
    encoder_name += '_numlayer_' + str(encoder_config['num_layer']) + '_embdim_' + str(encoder_config['emb_dim'])
    encoder_name +=  '_gnntype_' + str(encoder_config['gnn_type'])
    encoder_name += '_bs_' + str(dataset_config['batch_size']) + "_status_" + str(encoder_status)
    
    contrastive_config = config['contrastive']
    if not contrastive_config['aug_1']:
        aug_1 = None
    else:
        aug_1 = contrastive_config['aug_1']

    if not contrastive_config['aug_2']:
        aug_2 = None
    else:
        aug_2 = contrastive_config['aug_2']

    assert contrastive_config['emb_dim'] == encoder_config['emb_dim']
    graphcl = GNNGraphCL(contrastive_config['emb_dim'], aug_1=aug_1, aug_2=aug_2, tau=contrastive_config['tau'], device=device)
    
    evaluator_config = config['evaluator']
    if not evaluator_config['type']:
        print("Error no evaluator type")
        return
    elif evaluator_config['type'] == 'node_unsupervised':
        evaluator = NodeUnsupervised(dataset, train_loader, valid_loader, test_loader, clf_or_reg=task_type, device=device, config=evaluator_config)
        encs = evaluator.evaluate(learning_model=graphcl, encoder=encoder)
        save_encoder = encs[-1]

    if load_save_config["save_model"]:
        encoder_save_file = save_dir + encoder_name + ".pth"

        if os.path.exists(encoder_save_file):
            backup_file_name = encoder_save_file + ".bak-"+ now_time
            os.system(f'mv {encoder_save_file} {backup_file_name}')
        torch.save(save_encoder.state_dict(), encoder_save_file)


    
    

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
    # print("======")
    # test2(True)
    
    