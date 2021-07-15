import torch


print(torch.cuda.get_device_name(0))
x = torch.rand(5, 3)
print(x)