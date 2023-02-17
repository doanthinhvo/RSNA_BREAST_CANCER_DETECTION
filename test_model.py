import torch 
y = [torch.tensor([1., 2., 3.]), torch.tensor([4., 5., 6.])]
softmax_ed = [] 
for a in y:
    softmax_ed.append(torch.softmax(a, dim=-1)) # dim=-1 means the last dimension

print(y)
print(softmax_ed)