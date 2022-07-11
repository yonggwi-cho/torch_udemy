import torch
import numpy as np

a = torch.tensor([1,2,3])
print(a,type(a))

b = torch.tensor([[1,2],
                  [3,4]])
print(b)
print(b.size())

c = b.numpy()
print(c,type(c))

d = np.array([[1,2],[3,4]])
e = torch.from_numpy(d)
print(d,type(d))
print(e,type(e))

a = torch.tensor([[1,2,3],[4,5,6.]])
m = torch.mean(a)
print(m.item())

m = a.mean()
print(m.item())
print(a.mean(0))
