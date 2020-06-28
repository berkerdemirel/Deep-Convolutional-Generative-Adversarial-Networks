import torchvision
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import transforms
import torchvision.utils as vutils
import torch

import os
import numpy as np
import errno
from IPython import display
from matplotlib import pyplot as plt


class Discriminator(torch.nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1024, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Linear(1024 * 4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(-1, 1024 * 4) # flatten
        x = self.out(x) # sigmoid activation
        return x


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.linear = torch.nn.Linear(100, 256 * 4 * 4)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1)  # to resize the number of channels (convert it to image)
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        x = self.linear(x)  # increase dimensionality of latent vector
        x = x.view(x.shape[0], 256, 4, 4)  # reshape

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.out(x)  # apply tanh activation


def noise(size):
    n = Variable(torch.randn(size, 100))
    if torch.cuda.is_available():
        return n.cuda()
    return n


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.00, 0.02)


def real_data_target(size):
    data = Variable(torch.ones(size, 1).add_(-0.05))  # label smoothing targets
    if torch.cuda.is_available():
        return data.cuda()
    return data


def fake_data_target(size):
    data = Variable(torch.zeros(size, 1).add_(0.05))  # label smoothing targets
    if torch.cuda.is_available():
        return data.cuda()
    return data


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

num_batches = len(train_loader)

CONTINUE = False

generator = Generator()
discriminator = Discriminator()

if CONTINUE:
    generator.load_state_dict(torch.load(
        'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/data/models/G_initial_epoch_32'))
    discriminator.load_state_dict(torch.load(
        'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/data/models/D_initial_epoch_32'))
else:
    generator.apply(init_weights)
    discriminator.apply(init_weights)

# Enable CUDA if available
if torch.cuda.is_available():
    generator.cuda()
    discriminator.cuda()

# Optimizers
d_optimizer = Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
loss = nn.BCELoss()

# Number of epochs
num_epochs = 100

out_dir_model = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/data/models/'
out_dir_images = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/data/images'
home_dir = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/'
make_dir(home_dir)
make_dir(out_dir_images)
make_dir(out_dir_model)

num_test_samples = 16
test_noise = noise(num_test_samples)  # generated noise for testing generator's performance


def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad() # reset gradients

    prediction_for_real_data = discriminator(real_data)  # prediction of real data

    # calculate error and backpropagate
    error_real = loss(prediction_for_real_data, real_data_target(real_data.size(0)))
    error_real.backward()

    prediction_for_fake_data = discriminator(fake_data)  # prediction of fake data

    # calculate error and backpropagate
    error_fake = loss(prediction_for_fake_data, fake_data_target(real_data.size(0)))
    error_fake.backward()

    optimizer.step()  # update parameters

    return error_real + error_fake, prediction_for_real_data, prediction_for_fake_data


def train_generator(optimizer, fake_data):
    optimizer.zero_grad()  # reset gradients

    prediction = discriminator(fake_data)  # discriminator evaluates fake data

    # calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()

    optimizer.step()  # update parameters

    return error


for epoch in range(num_epochs):
    for n_batch, (real_batch, _) in enumerate(train_loader):

        real_data = Variable(real_batch)
        if torch.cuda.is_available():
            real_data = real_data.cuda()
        fake_data = generator(noise(real_data.size(0))).detach()  # generator

        # train discriminator
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        # train generator (if loss is high we fooled discriminator)
        fake_data = generator(noise(real_batch.size(0)))  # generate images from random noise
        g_error = train_generator(g_optimizer, fake_data)

        print("Discriminator real accuracy:{0}, Discriminator fake accuracy:{1}, "
              "epoch:{2}, {3}/{4} batches completed".
              format(d_pred_real.data.mean(), 1-d_pred_fake.data.mean(), epoch + 1, n_batch, num_batches))

        if n_batch % 100 == 0:  # save figures and models, display results
            display.clear_output(True)
            # Display Images
            test_images = generator(test_noise).data.cpu()
            number_of_rows = int(np.sqrt(num_test_samples))
            grid = vutils.make_grid(
                test_images, nrow=number_of_rows, normalize=True, scale_each=True)
            fig = plt.figure()

            plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
            plt.axis('off')

            fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(out_dir_images,
                                                             'G', epoch, n_batch))

            print('Epoch: [{}/{}], Batch Num: [{}/{}]'.format(
                epoch, num_epochs, n_batch, num_batches))

            print('Discriminator Loss: {:.4f}, Generator Loss: {:.4f}'.format(d_error.data.cpu().numpy(),
                                                                              g_error.data.cpu().numpy()))

            print('D(x): {:.4f}, D(G(z)): {:.4f}'.format(d_pred_real.data.mean(), d_pred_fake.data.mean()))

            torch.save(generator.state_dict(),
                       '{}/G_epoch_{}'.format(out_dir_model, epoch))
            torch.save(discriminator.state_dict(),
                       '{}/D_epoch_{}'.format(out_dir_model, epoch))
