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
    
def train_model(model,device,data_loader,loss_func,optimizer,num_epochs):
    train_loss, train_acc = [], []
    for epoch in range(num_epochs):
        runningLoss = 0.0
        correct = 0
        total = 0

        for images, labels, in data_loader:
            images, labels = images.to(device),labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_func(outputs,labels)
            loss.backward()

            optimizer.step()

            runningLoss += loss.item()

            _,predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = runningLoss / len(data_loader)
        epoch_acc = correct / total

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        if epoch_acc >= EPOCH_BREAK_ACCURACY:
            print(f"Model has reached {EPOCH_BREAK_ACCURACY} accuracy, stopping training")
            break

    return train_loss, train_acc

def test_model(model, data_loader, device=None):
    if device is None:
        device = torch.device('cpu')
    
    model.eval()#sets model to eval mode, dont want to change weights of the model
    test_loss = 0
    correct = 0

    data_len = len(data_loader.dataset)

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data) #retrieve predictions
            test_loss += functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(data_loader.dataset) #Calculating average loss
    accuracy = correct / data_len
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{data_len} ({100 *
    accuracy}%)')

    return accuracy