import torch

def get_device(config_gpu):
    if torch.cuda.is_available() and config_gpu != 'cpu':
        device = config_gpu
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    print("Running on:", device)

    return device