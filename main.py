import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchsummary import summary
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time

epochs = 100
batch_size = 64
latent_dim = 100
image_size = 28
lr = 0.0001

start_time = time.time()

transformation = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.Resize(image_size), transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])
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
            nn.ConvTranspose2d(in_channels=latent_dim, out_channels=256, kernel_size=(3, 3), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 512, kernel_size=(4, 4), stride=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 1, kernel_size=(2, 2), stride=(2, 2)),
            nn.BatchNorm2d(1),
            nn.ReLU(),

            nn.Tanh()

        )

    def forward(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 56, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(56, 224, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.BatchNorm2d(224),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(224, 448, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(448),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),

            nn.Conv2d(448, 1, kernel_size=(4, 4), stride=(1, 1), padding=0),
            nn.Sigmoid()

        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


generator = Generator().to(device)
discriminator = Discriminator().to(device)

print("Discriminator summary:")
summary(discriminator, (1, 28, 28))

print("Generator summary:")
summary(generator, (latent_dim, 1, 1))

lossFunction = nn.BCELoss()
generatorOptim = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
discriminatorOptim = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

x = torch.randn(1, 28, 28)
print("Random tensor: ", x.shape)
x_permute = x.permute(1, 2, 0)
plt.imshow(x.permute(1, 2, 0), cmap='gray')
plt.show()
print("Random tensor permute: ", x_permute.shape)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(m.bias.data, 0)


def show_images(image_tensor, size=(1, 28, 28)):
    flatten_image = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(flatten_image[:25], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze(), cmap='gray')
    plt.show()


def gen_noise(b_size):
    _generatedNoise = torch.randn(b_size, latent_dim, 1, 1, device=device)
    return _generatedNoise


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

display_step = 300
mean_generator_loss = 0
mean_discriminator_loss = 0

print("Starting Training Loop...")
for epoch in range(epochs):
    for real, _ in tqdm(data_loader):

        show = real.shape
        cur_batch_size = len(real)
        real = real.to(device)
        show_real = real.shape
        print(show_real, "\n", real)
        ## Update discriminator ##
        discriminatorOptim.zero_grad()
        disc_real_pred = discriminator(real).reshape(-1)
        disc_real_pred_shape = disc_real_pred.shape
        real_label = (torch.ones(cur_batch_size) * 0.9).to(device)
        real_label_size = real_label.shape
        # Get the discriminator's prediction on the real image and
        # calculate the discriminator's loss on real images
        disc_real_loss = lossFunction(disc_real_pred, real_label)
        # generate the random noise
        fake_noise = gen_noise(cur_batch_size)
        fake_noice_length = len(fake_noise)
        # generate the fake images by passing the random noise to the generator
        fake = generator(fake_noise)
        fake_shape = fake.shape
        # Get the discriminator's prediction on the fake images generated by generator
        disc_fake_pred = discriminator(fake.detach()).reshape(-1)
        fake_label = (torch.ones(cur_batch_size) * 0.1).to(device)
        # calculate the discriminator's loss on fake images
        disc_fake_loss = lossFunction(disc_fake_pred, fake_label)
        # Calculate the discriminator's loss by
        # accumulating the real and fake loss
        disc_loss = (disc_fake_loss + disc_real_loss)
        # Keep track of the average discriminator loss
        mean_discriminator_loss += disc_loss.item() / display_step
        # Update gradients
        disc_loss.backward()
        # Update optimizer
        discriminatorOptim.step()
        ## Update generator ##
        generatorOptim.zero_grad()

        # Get the discriminator's prediction on the fake images

        disc_fake_pred = discriminator(fake)
        real_label = (torch.ones(cur_batch_size)).to(device)
        #  Calculate the generator's loss.
        # the generator wants the discriminator to think that the
        # fake images generated by generator are real
        gen_loss = lossFunction(disc_fake_pred, real_label)
        # Backprop through the generator
        # update the gradients and optimizer.
        gen_loss.backward()
        generatorOptim.step()
        # Keep track of the average generator loss
        mean_generator_loss += gen_loss.item() / display_step
        ## Visualization code ##
        if iters % display_step == 0 and iters > 0:
            print(
                f"Epoch:{epoch} Step {iters}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            show_images(fake)
            show_images(real)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        iters += 1

        # OLD CODE !!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # # Output training stats
        # if i % 50 == 0:
        #     print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
        #           % (epoch, epochs, i, len(data_loader),
        #              errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        #
        # # Save Losses for plotting later
        # G_losses.append(errG.item())
        # D_losses.append(errD.item())
        #
        # # Check how the generator is doing by saving G's output on fixed_noise
        # if (iters % 500 == 0) or ((epoch == epochs - 1) and (i == len(data_loader) - 1)):
        #     with torch.no_grad():
        #         fake = generator(fixed_noise).detach().cpu()
        #     img_list.append(utils.make_grid(fake, padding=2, normalize=True))
        #
        # iters += 1

    if epoch % 50 == 0:
        torch.save(generator, 'Generator_epoch_{}.pth'.format(epoch))
        print('Model saved.')

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
