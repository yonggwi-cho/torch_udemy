import torch
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

mnist_train = MNIST("./data",train=True,download=True,transform=transforms.ToTensor())
mnist_test = MNIST("./data",train=False,download=True,transform=transforms.ToTensor())
print("# of train data =",len(mnist_train),", # of test data =",len(mnist_test))

img_size = 28
batch_size = 256
train_loader = DataLoader(mnist_train,batch_size=batch_size,shuffle=True)
test_loader  = DataLoader(mnist_test,batch_size=batch_size,shuffle=False)

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(img_size*img_size,1024) # fc
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,10)
    
    def forward(self,x):
        x = x.view(-1,img_size*img_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
net.cuda()
print(net)
