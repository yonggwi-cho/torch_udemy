# image generation by RNN

from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch

fmnist_data = FashionMNIST(root="./data",
                            train=True, download=True,
                            transform=transforms.ToTensor())
fmnist_classes = np.array(["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"])                        

print("# data :", len(fmnist_data))

n_image = 25
fmnist_loader = DataLoader(fmnist_data,batch_size=n_image,shuffle=True)
dataiter = iter(fmnist_loader)
images,labels = dataiter.next()

img_size = 28
plt.figure(figsize=(10,10))
for i in range(n_image):
    plt.subplot(5,5,i+1)
    plt.imshow(images[i].reshape(img_size,img_size),cmap="Greys_r")
    label = fmnist_classes[labels[i]]
    plt.title(label)
    plt.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)

#plt.show()

n_time = 14
n_in = img_size
n_mid = 256
n_out = img_size
n_sample_in_img = img_size-n_time

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

train_loader = DataLoader(dataset,batch_size=128,shuffle=True)

n_disp = 10
disp_data = FashionMNIST(root="./data",
                        )



