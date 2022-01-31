import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.utils as utils
import numpy as np
import time

epochs = 100
batch_size = 100
latent_dim = 100
image_size = 28
lr = 0.001

start_time = time.time()

transformation = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Resize(image_size)])
train_dataset = datasets.MNIST('mnist/', train=True, transform=transformation, download=True)
data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

plt.close("all")
for x, _ in data_loader:
    plt.imshow(x.numpy()[0][0], cmap='gray')
    plt.title('sample data')
    plt.show()
    print("sample batch data shape: ", x.shape)
    break


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.main = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=196, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(196),
            nn.ReLU(),

            nn.ConvTranspose2d(196, 392, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(392),
            nn.ReLU(),

            nn.ConvTranspose2d(392, 196, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(196),
            nn.ReLU(),

            nn.ConvTranspose2d(196, 1, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Tanh()

            # nn.ConvTranspose2d(196, 392, kernel_size=(4, 4), stride=(2, 2)),
            # nn.BatchNorm2d(392),
            # nn.ReLU(),
            #
            # nn.ConvTranspose2d(392, 196, kernel_size=(4, 4), stride=(2, 2)),
            # nn.BatchNorm2d(196),
            # nn.ReLU(),
            #
            # nn.ConvTranspose2d(196, 28, kernel_size=(4, 4), stride=(1, 1)),
            # nn.BatchNorm2d(28),
            # nn.ReLU(),
            #
            # nn.ConvTranspose2d(28, 1, kernel_size=(4, 4), stride=(1, 1)),
            # nn.BatchNorm2d(1),
            # nn.ReLU(),
            #
            # nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(1, 1)),
            #
            # nn.ConvTranspose2d(1, 1, kernel_size=(3, 3), stride=(1, 1)),
            #
            # nn.Tanh()

        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(28, 56, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(56),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(56, 28, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(28),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(28, 1, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Flatten(),
            nn.Linear(784, 1),
            nn.Sigmoid()

        )

    def forward(self, input):
        output = self.main(input)
        return output


generator = Generator().to(device)
discriminator = Discriminator().to(device)

print("Discriminator summary:")
summary(discriminator, (1, 28, 28))

print("Generator summary:")
summary(generator, (latent_dim, 1, 1))

lossFunction = nn.BCELoss()
generatorOptim = optim.Adam(generator.parameters(), lr=lr)
discriminatorOptim = optim.Adam(discriminator.parameters(), lr=lr)

x = torch.randn(1, 28, 28)
print("Random tensor: ", x.shape)
x_permute = x.permute(1, 2, 0)
plt.imshow(x.permute(1, 2, 0), cmap='gray')
plt.show()
print("Random tensor permute: ", x_permute.shape)


# generator1 = Generator()
# example_noise = torch.randn(1, latent_dim, 1, 1)
# fake = generator1(example_noise).squeeze().detach()
# plt.title("Example Image from Generator")
# plt.imshow(fake)
# plt.show()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def show_images(images):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    for index, image in enumerate(images):
        plt.subplot(sqrtn, sqrtn, index + 1)
        plt.imshow(image.reshape(28, 28), cmap='gray')

    plt.show()


generator.apply(weights_init)
discriminator.apply(weights_init)

fixed_noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
print("fixe_noise size", fixed_noise.shape)

real_label = 1.
fake_label = 0.

img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
# For each epoch
for epoch in range(epochs):
    # For each batch in the dataloader
    for i, data in enumerate(data_loader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        discriminator.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = discriminator(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = lossFunction(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        # Generate fake image batch with G
        fake = generator(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = discriminator(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = lossFunction(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        discriminatorOptim.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        generator.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = discriminator(fake).view(-1)
        # Calculate G's loss based on this output
        errG = lossFunction(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        generatorOptim.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, epochs, i, len(data_loader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(data_loader) - 1)):
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
            img_list.append(utils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    plt.clf()
    images_numpy = (fake.data.cpu().numpy() + 1.0) / 2.0
    show_images(images_numpy[:16])
    # if epoch % 50 == 0:
    #     torch.save(generator, 'Generator_epoch_{}.pth'.format(epoch))
    #     print('Model saved.')

print('Cost Time: {}s'.format(time.time() - start_time))
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig = plt.figure(figsize=(6, 6))
ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
