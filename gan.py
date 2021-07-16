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
transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,)),]
)
#data_iter = DataLoader(dataset = data, batch_size=batch_size)

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
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net_D(x)

#Hyperparameters etc 

device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4
z_dim = 2
image_dim = 2
batch_size = 8
num_epochs = 10

data = torch.from_numpy(data)
print(data.shape)
#data_iter = DataLoader(dataset = torch.transpose(data,0,1), batch_size=batch_size)

# Save the update

net_D = Discriminator(image_dim).to(device)
net_G = Generator(z_dim, image_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)

opt_disc = optim.Adam(net_D.parameters(), lr = lr)
opt_gen = optim.Adam(net_G.parameters(), lr = lr)
criterion = nn.BCELoss()
#writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
#writer_real = SummaryWriter(f"runs/GAN_MNIST/real")

step = 0
lst = []
for epoch in range(num_epochs):
    for batch_idx, (real) in enumerate(data):
        #print(real.shape)
        #print(type(real))
        real = real.view(-1, 2).to(device)
        batch_size = 50
        #print(real.shape)
        #print(real)
        real = real.float()
        #Train Discriminator : max log(D(real)) + log(1 - D(G(z)))
        noise = torch.randn(batch_size, z_dim).to(device)
        #print(noise)
        fake = net_G(noise)
        disc_real = net_D(real).view(-1)
        #print("disc_real")
        #print(disc_real)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = net_D(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake + lossD_real) / 2
        net_D.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()


        ## Train Generator min log(1 - D(G(z))) <--> max log(D(G(z)))
        output = net_D(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        net_G.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                    f"Epoch [{epoch}/{num_epochs}] \ "
                    f"Loss D: {lossD:.4f}, Loss G:{lossG:.4f}"
                    )
fake = fake.cpu().detach().numpy()                
print(fake)
print(fake[:,0])
plt.scatter(fake[:,(0)], fake[:,(1)])
plt.show()


