from sklearn import config_context
import datasets.download_datasets
import yaml
from torch_geometric.loader import DataLoader
from models.fine_tuning_models import GNNGraphPred
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr

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

def get_device(config_gpu):
    if torch.cuda.is_available() and config_gpu != 'cpu':
        device = config_gpu
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    print("Running on:", device)

    return device

def clf_training(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 > 0
        #Loss matrix
        criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")
        loss_mat = criterion(pred.double(), (y+1)/2)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()
        optimizer.step()
    
def reg_training(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        loss = torch.sum((pred-y)**2)/y.size(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    

def clf_evaluation(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list) #y_true.shape[1]

def reg_evaluation(model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy().flatten()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy().flatten()
    print(y_true.shape, y_scores.shape)
    mse = mean_squared_error(y_true, y_scores)
    cor = pearsonr(y_true, y_scores)[0]
    print(mse, cor)
    return mse, cor


def main():
    # set up
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)
    # device = get_device(config['gpu'])
    device = 'cpu'
    
    # set up dataset
    dataset_config = config['fine_tune_dataset']
    dataset_name = dataset_config['ogbg_dataset_name']
    dataset_path = dataset_config['ogbg_data_path']
    num_tasks, task_type, metric = get_num_type_metric_of_ogbg_dataset(dataset_name)
    dataset = datasets.download_datasets.get_ogbg_dataset(dataset_name=dataset_name, root=dataset_path)
    train_loader, valid_loader, test_loader = datasets.download_datasets.get_ogbg_dataset_loaders(dataset_name, batch_size=dataset_config['batch_size'], root=dataset_path)
    
    # set up model
    model_config = config['fine_tune_model']
    model = GNNGraphPred(model_config['num_layer'], model_config['emb_dim'], num_tasks, JK = model_config['JK'], 
                        drop_ratio = model_config['dropout_ratio'], graph_pooling = model_config['graph_pooling'], 
                        gnn_type = model_config['gnn_type'])
    # TODO: preloaded model?
    model.to(device)

    # set up optimizer
    # different learning rate for different part of GNN
    optimizer_config = config['fine_tune_optimizer']
    model_param_group = []
    model_param_group.append({"params": model.gnn.parameters()})
    model_param_group.append({"params": model.graph_pred_linear.parameters(), "lr":optimizer_config['lr']*optimizer_config['lr_scale']})
    optimizer = optim.Adam(model_param_group, lr=optimizer_config['lr'], weight_decay=optimizer_config['decay'])
    print(optimizer)

    # train, validate and test
    training_config = config['fine_tune_training']

    # training based on task type
    if task_type == 'clf':
        wait = 0
        best_auc = 0
        patience = 10
        for epoch in range(1, training_config['epoch']+1):
            print("====epoch " + str(epoch))
            
            clf_training(model, device, train_loader, optimizer)

            print("====Evaluation")
            if training_config['eval_train']:
                train_auc = clf_evaluation(model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_auc = 0
            val_auc = clf_evaluation(model, device, valid_loader)
            test_auc = clf_evaluation(model, device, test_loader)

            print("train: %f val: %f test: %f" %(train_auc, val_auc, test_auc))

            # Early stopping
            if np.greater(val_auc, best_auc):  # change for train loss
                best_auc = val_auc
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print('Early stop at Epoch: {:d} with final val auc: {:.4f}'.format(epoch, val_auc))
                    break

    elif task_type == 'reg':
        
        for epoch in range(1, training_config['epoch']+1):
            print("====epoch " + str(epoch))

            reg_training(model, device, train_loader, optimizer)

            print("====Evaluation")
            if  training_config['eval_train']:
                train_mse, train_cor = reg_evaluation(model, device, train_loader)
            else:
                print("omit the training accuracy computation")
                train_mse, train_cor = 0, 0
            val_mse, val_cor = reg_evaluation(model, device, valid_loader)
            test_mse, test_cor = reg_evaluation(model, device, test_loader)

            print("mse train: %f mse val: %f mse test: %f" %(train_mse, val_mse, test_mse))
            print("cor train: %f cor val: %f cor test: %f" %(train_cor, val_cor, test_cor))

if __name__ == '__main__':
    main()

    

