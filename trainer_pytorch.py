import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import pickle
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_dir = Path(r'CIFAR\cifar-10-batches-py')
batch = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']

train_batches = [unpickle(data_dir / batch[i]) for i in range(5)]
test_batch = unpickle(data_dir / 'test_batch')

X = np.array([i[b'data'] for i in train_batches], dtype='float').reshape(50000,3,32,32)/255
y = np.array([i[b'labels'] for i in train_batches], dtype='float').reshape(50000,1)
X_val = np.array(test_batch[b'data'], dtype='float').reshape(10000,3,32,32)/255
y_val = np.array(test_batch[b'labels'], dtype='float').reshape(10000,1)

batchsize=32
trainset = torch.utils.data.TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
testset = torch.utils.data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, drop_last=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, drop_last=True)

class GoodNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.Norm = nn.BatchNorm2d(3)
        self.Conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding='same'), nn.ReLU(), nn.BatchNorm2d(32))
        self.Conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding='same'), nn.ReLU(), nn.BatchNorm2d(64))
        self.Conv3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding='same'), nn.ReLU(), nn.BatchNorm2d(64))
        self.Pool = nn.MaxPool2d(2, 2)
        self.Conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, padding='same'), nn.ReLU(), nn.BatchNorm2d(128))
        self.Conv5 = nn.Sequential(nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(), nn.BatchNorm2d(128))
        self.Conv6 = nn.Sequential(nn.Conv2d(128, 128, 3, padding='same'), nn.ReLU(), nn.BatchNorm2d(128))
        self.Dense = nn.Sequential(nn.Linear(128*8*8, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Linear(1024, 10))
    def forward(self, x):
        x = self.Norm(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Pool(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)
        x = self.Pool(x)
        x = x.view(-1, 128*8*8)
        x = self.Dense(x)
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GoodNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


for epoch in range(10):
    correct = correct_test = 0
    for i, batch in enumerate(trainloader):
        X, y = batch
        X, y = X.to(device), y.to(device)
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y.squeeze())
        loss.backward()
        optimizer.step()
        correct += (torch.sum(y_pred.argmax(1)==y.squeeze())).int()
    with torch.no_grad():
        for j, batch in enumerate(testloader):
            model.eval()
            X_test, y_test = batch
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred_test = model(X_test)
            correct_test += (torch.sum(y_pred_test.argmax(1)==y_test.squeeze())).int()
    print(f'Training Accuracy: {correct/((i+1)*batchsize)*100:.2f} %')
    print(f'Testing Accuracy: {correct_test/((j+1)*batchsize)*100:.2f} % \n')

# Achieve ~75% accuracy after 10 epochs