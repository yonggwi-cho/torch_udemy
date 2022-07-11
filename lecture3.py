# dataloader

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self,size):
        super().__init__()
        self.size = size
        self.fc1 = nn.Linear(self.size*self.size,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,10)

    def forward(self,x):
        x = x.view(-1,self.size*self.size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import time

mnist_train = MNIST("./data",train=True,download=True, transform=transforms.ToTensor())
mnist_test = MNIST("./data",train=False,download=True, transform=transforms.ToTensor())

print("#train : ",len(mnist_train)," #test : ",len(mnist_test))

img_size = 28
batch_size = 256
n_epoch = 10
train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(mnist_test,batch_size=batch_size,shuffle=False)

net = Net(img_size)
#net.cuda()
#print(net)
print(1)


loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.01)

hist_loss_train = list()
hist_loss_test = list()

for i in range(n_epoch):
    net.train()
    loss_train = 0
    for j, (x,t) in enumerate(train_loader):
        x,t = x.cuda(),t.cuda()
        y = net(x)
        loss = loss_fnc(y,t)
        loss_train += loss.item()
        optmizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    hist_loss_train.append(loss_train)

    net.eval()
    loss_test = 0
    for j, (x,t) in enumerate(test_loader):
        x,t = x.cuda(),t.cuda()
        y = net(x)
        loss = loss_fnc(y,t)
        loss_test += loss.item()
    loss_test /= j+1
    hist_loss_test.append(loss_test)

    if i%1 == 0:
        print("#epoch",i,"Loss_Train:",loss_train,"Loss_Test:",loss_test)
