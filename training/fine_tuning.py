from sklearn import config_context
import datasets.fine_tuning.ogb.download_datasets as download_ogb
import yaml
from torch_geometric.loader import DataLoader
from method.gnn_models import GNNGraphPred
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import pearsonr
import os
import datetime
from utils import get_device

def clf_training(model, device, loader, optimizer):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = batch.y.view(pred.shape).to(torch.float64)

        #Whether y is non-null or not.
        is_valid = y**2 >= 0
        #Loss matrix
        criterion = torch.nn.BCEWithLogitsLoss(reduction = "none")
        loss_mat = criterion(pred.double(), y)
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
        #AUC is only defined when there is more than one class.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            is_valid = y_true[:,i]**2 >= 0
            # roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            roc_list.append(roc_auc_score(y_true[is_valid,i], y_scores[is_valid,i]))

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
    mse = mean_squared_error(y_true, y_scores)
    cor = pearsonr(y_true, y_scores)[0]
    print(mse, cor)
    return mse, cor


def main():
    # set up
    config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
    device = get_device(config['gpu'])
    torch.manual_seed(config['runseed'])
    np.random.seed(config['runseed'])
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # device = 'cpu'
    
    # set up dataset
    dataset_config = config['fine_tune_dataset']
    dataset_name = dataset_config['ogbg_dataset_name']
    print(f'dataset name: {dataset_name}')
    dataset_path = dataset_config['ogbg_data_path']
    num_tasks, task_type, metric = download_ogb.get_num_type_metric_of_ogbg_dataset(dataset_name)
    dataset = download_ogb.get_ogbg_dataset(dataset_name=dataset_name, root=dataset_path)
    train_loader, valid_loader, test_loader = download_ogb.get_ogbg_dataset_loaders(dataset_name, batch_size=dataset_config['batch_size'], root=dataset_path)

    # set up load and save directories
    load_save_config = config["load_save_fine_tune"]
    if load_save_config["load_model_dir"]:
        load_dir = 'results/' + load_save_config["load_model_dir"]
        load_file = load_dir + load_save_config["load_model_name"]
    else:
        load_file = None
    
    save_dir = 'results/fine_tuning/' + dataset_name + "/"
    if not os.path.exists(save_dir):
        os.system(f'mkdir -p {save_dir}')

    # set up model
    model_config = config['fine_tune_model']
    model = GNNGraphPred(model_config['num_layer'], model_config['emb_dim'], num_tasks, JK = model_config['JK'], 
                        drop_ratio = model_config['dropout_ratio'], graph_pooling = model_config['graph_pooling'], 
                        gnn_type = model_config['gnn_type'])
    
    if load_file:
        model.from_pretrained(load_file + ".pth")
        print('successfully load pretrained model!')
        model_status = "loaded"
    else:
        print('No pretrain! train from scratch!')
        model_status = "scratch"

    
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

    model_name = dataset_name + '_do_' + str(model_config['dropout_ratio']) + '_seed_' + str(config['runseed']) + '_JK_' + str(model_config['JK'])
    model_name += '_numlayer_' + str(model_config['num_layer']) + '_embdim_' + str(model_config['emb_dim'])
    model_name += '_graphpooling_' + str(model_config['graph_pooling']) + '_gnntype_' + str(model_config['gnn_type'])
    model_name += '_epoch_' + str(training_config['epoch']) + '_bs_' + str(dataset_config['batch_size']) + "_status_" + model_status
    model.to(device)

    txtfile = save_dir + model_name + ".txt"
    if os.path.exists(txtfile):
        backup_file_name = txtfile + ".bak-"+ now_time
        os.system(f'mv {txtfile} {backup_file_name}')

    # training based on task type
    if task_type == 'clf':
        with open(txtfile, "a") as myfile:
            myfile.write('epoch: train_auc val_auc test_auc\n')
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

            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_auc) + ' ' + str(val_auc) + ' ' + str(test_auc) + "\n")

            print("train auc: %f val auc: %f test auc: %f" %(train_auc, val_auc, test_auc))

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
        with open(txtfile, "a") as myfile:
            myfile.write('epoch: train_mse train_cor val_mse val_cor test_mse test_cor\n')
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

            with open(txtfile, "a") as myfile:
                myfile.write(str(int(epoch)) + ': ' + str(train_mse) + ' ' + str(train_cor) + ' ' + str(val_mse) + ' ' + str(val_cor) + ' ' + str(test_mse) + ' ' + str(test_cor) + "\n")
                
            print("mse train: %f mse val: %f mse test: %f" %(train_mse, val_mse, test_mse))
            print("cor train: %f cor val: %f cor test: %f" %(train_cor, val_cor, test_cor))

    if load_save_config["save_model"]:
        model_save_file = save_dir + model_name + ".pth"
        if os.path.exists(model_save_file):
            backup_file_name = model_save_file + ".bak-"+ now_time
            os.system(f'mv {model_save_file} {backup_file_name}')
        torch.save(model.gnn.state_dict(), model_save_file)

if __name__ == '__main__':
    main()

    

