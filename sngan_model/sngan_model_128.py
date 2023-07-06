# DCGAN-like generator and discriminator
from torch import nn
import torch.nn.functional as F

from sngan_model.spectral_normalization import SpectralNorm

channels = 1
leak = 0.1
w_g = 4

class Generator(nn.Module):
  def __init__(self, z_dim):
    super(Generator, self).__init__()
    self.z_dim = z_dim
    self.model = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d( z_dim, 1024, 4, 1, 0, bias=False),
      nn.BatchNorm2d(1024),
      nn.ReLU(True),
      
      nn.ConvTranspose2d( 1024, 512, 4, 2, 1, bias=False),
      nn.BatchNorm2d(512),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
      nn.BatchNorm2d(256),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      # state size. (ngf) x 32 x 32
      nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
      nn.Tanh()
    )
  def forward(self, z):
    return self.model(z.view(-1, self.z_dim, 1, 1))

class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 4, stride=2, padding=(1,1)))
    #self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1,1)))
    self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1,1)))
    #self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1,1)))
    self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=(1,1)))
    self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1,1)))
    self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=(1,1)))
    self.conv8 = SpectralNorm(nn.Conv2d(512, 1024, 4, stride=2, padding=(1,1)))
    self.fc = SpectralNorm(nn.Linear(w_g * w_g * 1024, 1))

  def forward(self, x):
    m = x
    m = nn.LeakyReLU(leak)(self.conv1(m))
    #m = nn.LeakyReLU(leak)(self.conv2(m))
    m = nn.LeakyReLU(leak)(self.conv3(m))
    #m = nn.LeakyReLU(leak)(self.conv4(m))
    m = nn.LeakyReLU(leak)(self.conv5(m))
    m = nn.LeakyReLU(leak)(self.conv6(m))
    m = nn.LeakyReLU(leak)(self.conv7(m))
    m = nn.LeakyReLU(leak)(self.conv8(m))

    return self.fc(m.view(-1,w_g * w_g * 1024))

