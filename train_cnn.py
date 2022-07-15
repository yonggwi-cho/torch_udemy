from turtle import down
from matplotlib import scale
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

affine = transforms.RandomAffine([-15,15],scale=(0.8,1.2))
flip = transforms.RandomHorizontalFlip(p=0.5)
normalize = transforms.Normalize((0.0,0.0,0.0),(1.0,1.0,1.0))
to_tensor = transforms.ToTensor()

transform_train = transforms.Compose([affine,flip,to_tensor,normalize])
transform_test = transforms.Compose([to_tensor,normalize])

cifar10_train = CIFAR10("./data",train=True,download=True,transform=transform_train)
cifar10_test = CIFAR10("./data",train=False,download=True,transform=transform_test)

batch_size = 64
train_loader = DataLoader(cifar10_train,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(cifar10_test,batch_size=batch_size,shuffle=False)

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1   = nn.Linear(16*5*5,256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(16*5*5,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = self.dropout(x)
        x = self.fc2(x)
        return x 

net = Net()
net.cuda(1)
n_epoch = 100

loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

hist_loss_train = list()
hist_loss_test = list()

x_test,t_test = iter(test_loader).next()
x_test,t_test = x_test.cuda(1),t_test.cuda(1)

for i in range(n_epoch):
    net.train()
    loss_train = 0
    for j, (x,t) in enumerate(train_loader):
        x,t = x.cuda(1),t.cuda(1)
        y = net(x)
        loss = loss_fnc(y,t)
        loss_train += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    loss_train /= j+1
    hist_loss_train.append(loss_train)

    if i%1 == 0 :
        print("Epoch:",i,"Loss_Train:",loss_train)

    net.eval()
    y_test = net(x_test)
    loss_test = loss_fnc(y_test,t_test).item()
    hist_loss_test.append(loss_test)


# loss plot
plt.plot(range(len(hist_loss_train)),hist_loss_train,label="Train")
plt.plot(range(len(hist_loss_test)),hist_loss_test,label="Test")
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("./result/loss_cnn.png")
plt.show()

correct =  0.0
total = 0.0
net.eval()
for i, (x,t) in enumerate(test_loader):
    x,t = x.cuda(),t.cuda()
    y = net(x)
    correct += (y.argmax(i)==t).sum().item()
    total += len(x)
print("correct rate : ",str(correct/total*100)+"%")


for key in net.state_dict():
    print(key,":",net.state_dict()[key].size())
print(net.state_dict()["conv1.weigh"][0])

torch.save(net.state_dict(),"./result/model_cnn.pth")



