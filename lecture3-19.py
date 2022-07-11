import torch

x = torch.ones(2,3,requires_grad=True)
print(x)

y = x + 2
print(y)
print(y.grad_fn)

z  = y *3 
print(z)

out = z.mean()
print(out)

# calc grad
a = torch.tensor([1.0],requires_grad=True)
b = a * 2
b.backward()
print(a.grad)

# more complex examole
def calc(a):
    b = a*2 +1
    c = b*b
    d = c/(c+2)
    e = d.mean()
    return e

x = [1.0,2.0,3.0]
x = torch.tensor(x,requires_grad=True)
y = calc(x)
y.backward()
print(y)
print(x.grad.tolist())
