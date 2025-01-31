import torch
from torch.utils.data import DataLoader
from torch.optim import RMSprop
from torchvision import datasets
from model import CNN, train_model, test_model
from lib.util import header, plot_distribution, double_plot, normalization_transform

SESSION_1_EPOCH_COUNT = 3
SESSION_2_EPOCH_COUNT = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000

def det_device_config(train_kwargs,test_kwargs):
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    if use_cuda:
        print('using cuda device')
        device = torch.device("cuda")
    elif use_mps:
        print('using mps')
        device = torch.device("mps")
    else:
        print('using cpu')
        device = torch.device('cpu')
    
    if use_cuda:
        cuda_kwargs = {'num_workers':1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
        
    return device, train_kwargs, test_kwargs