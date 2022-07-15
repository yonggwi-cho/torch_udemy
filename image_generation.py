# image generation by RNN

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch import optim

class Net(nn.Module):
    def __init__(self,n_in,n_mid,n_out) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=n_in,
            hidden_size=n_mid,
            batch_first=True,
        )
        self.fc = nn.Linear(n_mid,n_out)

    def forward(self,x):
        y_rnn,(h,c) = self.rnn(x,None)
        y = self.fc(y_rnn[:,-1,:])
        return y
    

fmnist_data = FashionMNIST(root="./data",
                            train=True, download=True,
                            transform=transforms.ToTensor())
fmnist_classes = np.array(["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])                        

print("# data :", len(fmnist_data))

n_epoch = 100
n_image = 25
img_size = 28
n_time = 14
n_disp = 10
n_in = img_size
n_mid = 256
n_out = img_size
n_sample_in_img = img_size-n_time

fmnist_loader = DataLoader(fmnist_data,batch_size=n_image,shuffle=True)
dataiter = iter(fmnist_loader)
images,labels = dataiter.next()

net = Net(n_in,n_mid,n_out)
net.cuda(1)
print(net)

"""
plt.figure(figsize=(10,10))
for i in range(n_image):
    plt.subplot(5,5,i+1)
    plt.imshow(images[i].reshape(img_size,img_size),cmap="Greys_r")
    label = fmnist_classes[labels[i]]
    plt.title(label)
    plt.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
plt.show()
"""

dataloader = DataLoader(fmnist_data,batch_size=len(fmnist_data),shuffle=False)
dataiter = iter(dataloader)
train_images,train_labels = dataiter.next()
train_images = train_images.reshape(-1,img_size,img_size)

n_sample = len(train_images) * n_sample_in_img

input_data = np.zeros((n_sample,n_time,n_in))
correct_data = np.zeros((n_sample,n_out))

for i in range(len(train_images)):
    for j in range(n_sample_in_img):
        sample_id = i*n_sample_in_img+ j
        input_data[sample_id] = train_images[i,j:j+n_time]
        correct_data[sample_id] = train_images[i,j+n_time]

input_data = torch.tensor(input_data,dtype=torch.float)
correct_data = torch.tensor(correct_data,dtype=torch.float)
dataset = torch.utils.data.TensorDataset(input_data,correct_data)

# train
train_loader = DataLoader(dataset,batch_size=128,shuffle=True)
loss_fnc = nn.MSELoss()
optimizer = optim.Adam(net.parameters())

# validation
disp_data = FashionMNIST(root="./data",
                        train=False, download=True,
                        transform=transforms.ToTensor())
disp_loader = DataLoader(disp_data,batch_size=n_disp,shuffle=False)
dataiter = iter(disp_loader)
disp_imgs,labels = dataiter.next()
disp_imgs = disp_imgs.reshape(-1,img_size,img_size)

plt.figure(figsize=(20,2))

def generate_images(save=False):
    for i in range(n_disp):
        ax = plt.subplot(2,n_disp,i+1)
        plt.imshow(disp_imgs[i],cmap="Greys_r",vmin=0.0,vmax=1.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_title("Original",x=2)

    gen_imgs = disp_imgs.clone()
    for i in range(n_disp):
        for j in range(n_sample_in_img):
            x = gen_imgs[i,j:j+n_time].reshape(-1,n_time,img_size)
            x = x.cuda(1)
            gen_imgs[i,j+n_time] = net(x)[0]
        ax = plt.subplot(2,n_disp,n_disp+i+1)
        plt.imshow(gen_imgs[i].detach(),cmap="Greys_r",vmin=0.0,vmax=1.0)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    ax.set_title("Generated",x=2)
    
    if save == False :
        plt.pause(1)
        plt.clf()
    elif save == True :
        plt.savefig("image_LSTM.png")
    
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

    if i%1 == 0 :
        print("Epoch:",i,"Loss_Train:",loss_train)
        generate_images()
    
generate_images(save=True)

