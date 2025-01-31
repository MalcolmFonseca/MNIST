import torch
from torch import nn
from torch.nn import functional

#CONFIG
#target accuracy, stop training when reached
EPOCH_BREAK_ACCURACY = .995
#num of imgs being tested
TEST_BATCH_SIZE = 1000

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        #convolutional layers
        self.conv1 == nn.Conv2d(1,32,3,1)
        self.conv2 == nn.Conv2d(32,64,3,1)

        self.dropout1 = nn.Dropout(0.25) #drops 25% neurons
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(9216,128)
        self.fc1 = nn.Linear(128,10)

    def forward(self,x):
        x = self.conv1(x)

        x = functional.relu(x)
        x = self.conv2(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x,2)

        x = self.dropout1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2
        return x