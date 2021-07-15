import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

X = np.random.normal(0.0,1,(1000,2))
A = np.array([[1,2],[-0.1, 0.5]])
b = np.array([1,2])
data = np.dot(X,A) + b

plt.scatter(data[:100,(0)], data[:100,(1)])
plt.show()
print(f'The covariance matrix is\n{np.dot(A.T, A)}')

batch_size = 8
data_iter = DataLoader(dataset = data, batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Generator 
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.net_G = nn.Sequential(
            nn.Linear(z_dim, 2),
        )
    def forward(self, x):
        return self.net_G(x)

class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.net_D = nn.Sequential(
            nn.Linear(img_dim, 5),
            nn.tanh(),
            nn.Linear(5, 3),
            nn.tanh(),
            nn.Linear(3,1),
        )
    def forward(self, x):
        return self.net_D(x)

# Save the update

net_D = Discriminator(image_dim).to(device)
net_G = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

opt_disc = optim.Adam(disc.parameters(), lr = lr)
opt_gen = optim.Adam(gen.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0


