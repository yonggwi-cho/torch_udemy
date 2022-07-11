import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=64*2,
            batch_first=True,
        )
        self.fc = nn.Linear(64*2,1)

    def forward(self,x):
        y_rnn, h = self.rnn(x,None)
        y = self.fc(y_rnn[:,-1,:])
        return y

sin_x = np.linspace(-2*np.pi,2*np.pi)
sin_y = np.sin(sin_x) + 0.1*np.random.randn(len(sin_x))
plt.plot(sin_x,sin_y)
plt.savefig("random_sin.png")
plt.clf()

n_epoch = 5000
n_time = 10
n_sample = len(sin_x) - n_time

input_data = np.zeros((n_sample,n_time,1))
correct_data = np.zeros((n_sample,1))

for i in range(n_sample):
    input_data[i] = sin_y[i:i+n_time].reshape(-1,1)
    correct_data[i] = sin_y[i+n_time:i+n_time+1]

input_data = torch.tensor(input_data,dtype=torch.float)
correct_data = torch.tensor(correct_data,dtype=torch.float)
dataset = torch.utils.data.TensorDataset(input_data,correct_data)

train_loader = DataLoader(dataset,batch_size=8,shuffle=True)

net = Net()
net.cuda(1)
print(net)

loss_fnc = nn.MSELoss()

optimizer = optim.SGD(net.parameters(),lr=0.01)

hist_loss_train = list()
hist_loss_test = list()

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

    if i%2 == 0:
        print("#epoch",i,"Loss_Train:",loss_train)
        predicted = list(input_data[0].reshape(-1))
        
        for i in range(n_sample):
            x = torch.tensor(predicted[-n_time:]).cuda(1)
            x = x.reshape(1,n_time,1)
            y = net(x)
            predicted.append(y[0].item())

        plt.plot(range(len(sin_y)),sin_y,label="Correct")
        plt.plot(range(len(predicted)),predicted,label="Predicted")
        plt.legend()
        plt.pause(2)
        plt.clf()

plt.plot(range(len(sin_y)),sin_y,label="Correct")
plt.plot(range(len(predicted)),predicted,label="Predicted")
plt.legend()
plt.savefig("predicted.pnv")