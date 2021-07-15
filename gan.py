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
print(X.shape)
A = np.array([[1,2],[-0.1, 0.5]])
b = np.array([1,2])
data = np.dot(X,A) + b
print(data.shape)

plt.scatter(data[:100,(0)], data[:100,(1)])
plt.show()
print(f'The covariance matrix is\n{np.dot(A.T, A)}')

batch_size = 8
data_iter = DataLoader(dataset = data, batch_size=batch_size)

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
            nn.Tanh(),
            nn.Linear(5, 3),
            nn.Tanh(),
            nn.Linear(3,1),
        )
    def forward(self, x):
        return self.net_D(x)

#Hyperparameters etc 

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1
batch_size = 32
num_epochs = 50

# Save the update

net_D = Discriminator(image_dim).to(device)
net_G = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

opt_disc = optim.Adam(net_D.parameters(), lr = lr)
opt_gen = optim.Adam(net_G.parameters(), lr = lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, ).to(device)
        batch_size = real.shape[0]

        #Train Discriminator : max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = net_G(noise)
        disc_real = net_D(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = net_D(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()


        ## Train Generator min log(1 - D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()




