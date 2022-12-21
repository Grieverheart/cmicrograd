import torch

x = torch.Tensor([-4.0]).double()
x.requires_grad = True
z = 2 * x + 2 + x
q = z.relu() + z * x
h = (z * z).relu()
y = h + q + q * x
y.backward()

print(y.data.item(), x.grad.item())
print()

a = torch.Tensor([-4.0]).double()
b = torch.Tensor([2.0]).double()
a.requires_grad = True
b.requires_grad = True
c = a + b
d = a * b + b**3
c = c + c + 1
c = c + 1 + c + (-a)
d = d + d * 2 + (b + a).relu()
d = d + 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g = g + 10.0 / f
g.backward()
apt, bpt, gpt = a, b, g

tol = 1e-6
# forward pass went well
print(gpt.data.item())
# backward pass went well
print(apt.grad.item())
print(bpt.grad.item())
