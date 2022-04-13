from training.pre_training import pre_train
from training.fine_tuning import fine_tune
import sys
import argparse
import yaml

def main(config):
    # first pre_train and get the resulting model path
    model_dir, model_file_name, pre_train_config = pre_train(config)
    # then fine_tune with augmentations
    model_dir, model_file_name, fine_tune_config = fine_tune(config, model_dir, model_file_name, pre_train_config=pre_train_config)

if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "-c", "--config", # default config file
        type=str
    )
    cli.add_argument(
        "-p", "--pretraindataset", # pretrain datasets
        nargs="*",
        type=str
    )
    cli.add_argument(
        "-f", "--finetunedataset", # fine tune datasets
        nargs="*",
        type=str
    )

    args = cli.parse_args()
    default_config_file = args.config
    path_to_config = f'config/{default_config_file}.yaml'
    config = yaml.load(open(path_to_config, "r"), Loader=yaml.FullLoader)

    for pre_train_dataset, fine_tune_dataset in zip(args.pretraindataset, args.finetunedataset):
        config['pre_train_dataset']['pubchem_dataset_name'] = pre_train_dataset
        config['fine_tune_dataset']['ogbg_dataset_name'] = fine_tune_dataset
        # first train non-scaffold aware
        print(f'==========================NON SCAFFOLD=========================')
        config['contrastive']['aug_1'] = 'maskN'
        config['contrastive']['aug_2'] = 'maskN'
        model_dir, model_file_name, pre_train_config = pre_train(config)
        config['load_save_fine_tune']['load_model_dir'] = f'pre_training/{pre_train_dataset}'
        config['load_save_fine_tune']['load_model_name'] = model_file_name
        fine_tune(config)
        # train scaffold aware 
        print(f'==========================SCAFFOLD=========================')
        config['contrastive']['aug_1'] = 'maskNScaffold'
        config['contrastive']['aug_2'] = 'maskNScaffold'
        model_dir, model_file_name, pre_train_config = pre_train(config)
        config['load_save_fine_tune']['load_model_dir'] = f'pre_training/{pre_train_dataset}'
        config['load_save_fine_tune']['load_model_name'] = model_file_name
        fine_tune(config)
        # try:
        #     config['pre_train_dataset']['pubchem_dataset_name'] = pre_train_dataset
        #     config['fine_tune_dataset']['ogbgb_dataset_name'] = fine_tune_dataset
        #     # first train non-scaffold aware
        #     print(f'==========================NON SCAFFOLD=========================')
        #     config['contrastive']['aug_1'] = 'maskN'
        #     config['contrastive']['aug_2'] = 'maskN'
        #     model_dir, model_file_name, pre_train_config = pre_train(config)
        #     config['load_save_fine_tune']['load_model_dir'] = f'pre_training/{pre_train_dataset}'
        #     config['load_save_fine_tune']['load_model_name'] = model_file_name
        #     fine_tune(config)
        #     # train scaffold aware 
        #     print(f'==========================SCAFFOLD=========================')
        #     config['contrastive']['aug_1'] = 'maskNScaffold'
        #     config['contrastive']['aug_2'] = 'maskNScaffold'
        #     model_dir, model_file_name, pre_train_config = pre_train(config)
        #     config['load_save_fine_tune']['load_model_dir'] = f'pre_training/{pre_train_dataset}'
        #     config['load_save_fine_tune']['load_model_name'] = model_file_name
        # except:
        #     print(f'error occured while pretraining on {pre_train_dataset} and finetuning on {fine_tune_dataset}')
    

    # template config
    # ['fine_tune_dataset']['ogbgb_dataset_name'] = fine_tune_dataset_name
    # ['pre_train_dataset']['pubchem_dataset_name'] = pre_train_dataset_name
    # ['contrastive']['aug_1'] = 
    # ['contrastive']['aug_2'] = 
    # ['load_save_fine_tune']['load_model_dir'] = f'pre_training/{pre_train_dataset_name}'
    # ['load_save_fine_tune']['load_model_name'] = pre_train_model_fie_name