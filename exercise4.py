# Lecture 4 CNN
# Random Erase

from random import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim
import time
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,256)
        self.dropput = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropput(x)
        x = self.fc2(x)
        return x

cifar10_data = CIFAR10(root="./data",train=True,download=True, transform=transforms.ToTensor())
cifar10_classes = np.array(["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"])

print("#train : ",len(cifar10_data)," #test : ",len(cifar10_classes))

n_epoch = 100
n_image = 25
cifar10_loader = DataLoader(cifar10_data,batch_size=n_image,shuffle=True)
dataiter = iter(cifar10_loader)
image,labels = dataiter.next()

plt.figure(figsize=(10,10))
for i in range(n_image):
    plt.subplot(5,5,i+1)
    plt.imshow(np.transpose(image[i],(1,2,0)))
    label = cifar10_classes[labels[i]]
    plt.title(label)
    plt.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)

#plt.show()

# learning by CNN
# preprocessing 
affine = transforms.RandomAffine([-15,15],scale=(0.8,1.2))
flip = transforms.RandomHorizontalFlip(p=0.5) 
normalize = transforms.Normalize((0.0,0.0,0.0),(1.0,1.0,1.0)) # RGB
to_tensor = transforms.ToTensor()
random_erase = transforms.RandomErasing(p=0.5)

transform_train = transforms.Compose([affine,flip,to_tensor,random_erase,normalize])
transform_test = transforms.Compose([to_tensor,normalize])
cifar10_train = CIFAR10("./data",train=True,download=True,transform=transform_train)
cifar10_test = CIFAR10("./data",train=False,download=True,transform=transform_test)

# setting for dataloader
batch_size = 64
train_loader = DataLoader(cifar10_train,batch_size=batch_size,shuffle=True)
test_loader  = DataLoader(cifar10_test,batch_size=len(cifar10_test),shuffle=False)

net = Net()
net.cuda(1)
print(net)

loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

hist_loss_train = list()
hist_loss_test = list()

x_test, t_test  = iter(test_loader).next()
x_test, t_test  = x_test.cuda(1),t_test.cuda(1)

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

    net.eval()
    y_test = net(x_test)
    loss_test = loss_fnc(y_test,t_test).item()
    hist_loss_test.append(loss_test)
    
    if i%1 == 0:
        print("#epoch",i,"Loss_Train:",loss_train,"Loss_Test:",loss_test)


plt.plot(range(len(hist_loss_train)),hist_loss_train,"r-")
plt.plot(range(len(hist_loss_test)),hist_loss_test,"b--")
plt.legend()

plt.xlabel("Epoch")
plt.ylabel("Error")
plt.show()

correct = 0
total = 0
net.eval()
for i,(x,t) in enumerate(test_loader):
    x,t = x.cuda(1), t.cuda(1)
    y = net(x)
    correct += (y.argmax(1)==t).sum().item()
    total += len(x)
print("correct rate : ",str(correct/total*100)+"%")