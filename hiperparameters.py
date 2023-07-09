from datetime import date
import torch, os

## GENERALS
# Path of input images
directory = os.getcwd()
directory = os.path.dirname(directory)
data_path = os.path.join(directory,'Datasets', 'Ultrasonido') 
print(data_path)
# Datetime
today = date.today()
today = today.strftime("%B %d, %Y")
# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if (cuda and ngpu > 0) else "cpu")
# Number of workers for dataloader
workers = 1 if cuda else 2
# Batch size during training
batch_size = 32
# Number of class images
n_class = 2
# Spatial size of training images. All images will be resized to this size using a transformer.
image_size = 128
# Number of channels in the training images. For color images this is 3
nc = 1
# Size of z latent vector (i.e. size of generator input)
nz = 100
# Number of training epochs
num_epochs = 600
# Learning rate for optimizers
lr = 0.00015
# Beta hyperparam for Adam optimizers
beta1 = 0.3
beta2 = 0.75
# Default start epoch
start_epoch = 0

## SNGAN
# Iters for discriminator
disc_iters = 5

## VALIDATION FID
# Evaluation each n epochs
interval_evaluation = 5
