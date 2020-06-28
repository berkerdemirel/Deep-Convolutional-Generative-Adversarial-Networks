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
                in_channels=3, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(
                in_channels=512, out_channels=1, kernel_size=4,
                stride=2, padding=0, bias=False
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print("x",x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Generator(torch.nn.Module):

    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=100, out_channels=512, kernel_size=4,
                stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=256, out_channels=128, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128, out_channels=64, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=3, kernel_size=4,
                stride=2, padding=1, bias=False
            )
        )
        self.out = torch.nn.Tanh()

    def forward(self, x):
        x = x.view(-1, 100, 1, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

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
    data = Variable(torch.ones(size, 1).add_(-0.02))  # label smoothing targets
    if torch.cuda.is_available():
        return data.cuda()
    return data


def fake_data_target(size):
    data = Variable(torch.zeros(size, 1).add_(0.02))  # label smoothing targets
    if torch.cuda.is_available():
        return data.cuda()
    return data


def make_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


data_set_dir = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/dataset'
# data_set_dir = './dataset/'
train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


batch_size = 100

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transformation)

train_sets = [[],[],[],[],[],[],[],[],[],[]]
print("Splitting the training set")
for i in range(len(train_set)):
    train_sets[train_set[i][1]].append(train_set[i][0])

num_batches = len(train_sets[0]) // batch_size

CONTINUE = True


generators = []
discriminators = []
for i in range(10):
    generator = Generator()
    generators.append(generator)
    disc = Discriminator()
    discriminators.append(disc)

if CONTINUE:
    print("Loading models...")
    for i in range(10):
        generators[i].load_state_dict(torch.load(
            'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/cifar10_last_model_part_1_home/data/models/'
            'G_epoch_49_class_{0}_initial'.format(i)))
        discriminators[i].load_state_dict(torch.load(
            'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/cifar10_last_model_part_1_home/data/models/'
            'D_epoch_49_class_{0}_initial'.format(i)))
else:
    for i in range(10):
        generators[i].apply(init_weights)
        discriminators[i].apply(init_weights)

# Enable CUDA if available
if torch.cuda.is_available():
    print("Working on CUDA")
    for i in range(10):
        generators[i].cuda()
        discriminators[i].cuda()
else:
    print("Working on CPU")

# Optimizers
g_optimizers = []
d_optimizers = []
for i in range(10):
    g_optimizer = Adam(generators[i].parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizers.append(g_optimizer)
    d_optimizer = Adam(discriminators[i].parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizers.append(d_optimizer)
# Loss function
loss = nn.BCELoss()

# Number of epochs
num_epochs = 50

out_dir_model = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/data/models/'
out_dir_images = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/data/images'
home_dir = 'C:/Users/berkerdemirel/PycharmProjects/cs515_gan_proj/home/'
make_dir(home_dir)
make_dir(out_dir_images)
make_dir(out_dir_model)

num_test_samples = 16
test_noise = noise(num_test_samples)  # generated noise for testing generator's performance


def train_discriminator(optimizer, real_data, fake_data, discriminator):
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


def train_generator(optimizer, fake_data, discriminator):
    optimizer.zero_grad()  # reset gradients

    prediction = discriminator(fake_data)  # discriminator evaluates fake data

    # calculate error and backpropagate
    error = loss(prediction, real_data_target(prediction.size(0)))
    error.backward()

    optimizer.step()  # update parameters

    return error


for epoch in range(num_epochs):
    for batch_num in range(num_batches):
        for class_id in range(10):
            real_batch = train_sets[class_id][batch_num*batch_size:(batch_num+1)*batch_size]
            r_b = torch.Tensor(100, 3, 64, 64)
            torch.cat(real_batch, out= r_b)
            real_data = Variable(r_b.view(100,3,64,64))

            if torch.cuda.is_available():
                real_data = real_data.cuda()
            fake_data = generators[class_id](noise(real_data.size(0))).detach()  # generator

            # train discriminator
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizers[class_id], real_data, fake_data, discriminators[class_id])

            # train generator (if loss is high we fooled discriminator)
            fake_data = generators[class_id](noise(real_data.size(0)))  # generate images from random noise
            g_error = train_generator(g_optimizers[class_id], fake_data, discriminators[class_id])

            print("Discriminator real accuracy:{0}, Discriminator fake accuracy:{1}, "
                  "epoch:{2}, {3}/{4} batches completed".
                  format(d_pred_real.data.mean(), 1-d_pred_fake.data.mean(), epoch + 1, batch_num, num_batches))

        if batch_num == 0 or (batch_num+1) % 25 == 0:  # save figures and models, display results
            display.clear_output(True)
            # Display Images
            for i in range(10):
                test_images = generators[i](test_noise).data.cpu()
                number_of_rows = int(np.sqrt(num_test_samples))
                grid = vutils.make_grid(
                    test_images, nrow=number_of_rows, normalize=True, scale_each=True)
                fig = plt.figure()

                plt.imshow(np.moveaxis(grid.numpy(), 0, -1))
                plt.axis('off')

                fig.savefig('{}/{}_epoch_{}_batch_{}_class_{}.png'.format(out_dir_images,
                                                                 'G', epoch, batch_num, i))

                torch.save(generators[i].state_dict(),
                           '{}/G_epoch_{}_class_{}'.format(out_dir_model, epoch, i))
                torch.save(discriminators[i].state_dict(),
                          '{}/D_epoch_{}_class_{}'.format(out_dir_model, epoch, i))
