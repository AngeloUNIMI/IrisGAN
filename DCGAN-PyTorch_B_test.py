#Imports
import argparse
import os
import numpy as np
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from PIL import Image

torch.cuda.empty_cache()


#------------------------------------------------------------------------PARAMS
#Directories
#modelDir = directory where the model is saved
modelDir = './models/'
outDir = './images_generated/'

#Create directory
os.makedirs(outDir, exist_ok=True)

#Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--n_images', type=int, default=1000, help='number of images to generate')
parser.add_argument('--batches', type=int, help='model chosen')
parser.add_argument('--batch_size', type=int, default=2, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_sizeH', type=int, default=64, help='height size of each image dimension')
parser.add_argument('--img_sizeW', type=int, default=512, help='width size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
opt = parser.parse_args()
print(opt)

#Models
#batches = 20000;
modelFileName = "model_save_%d.pth" % opt.batches


#-------------------------------------------------------------------Enable CUDA
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


#-----------------------------------------------------------------------CLASSES
#Weight initialization
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

#Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_sizeH = opt.img_sizeH // 4
        self.init_sizeW = opt.img_sizeW // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_sizeH*self.init_sizeW))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_sizeH, self.init_sizeW)
        img = self.conv_blocks(out)

        #print("img uscita da forward: ")
        #print(img.shape)

        return img

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_sizeH = opt.img_sizeH // 2**4
        ds_sizeW = opt.img_sizeW // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(128*ds_sizeH*ds_sizeW, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


#---------------------------------------------------------------------FUNCTIONS
def load_dataset():
    data_path = imgDir
    
    if (opt.channels == 1):
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transforms.Compose([
                    ###################
                    transforms.Grayscale(1),
                    ###################                
                    transforms.Resize((opt.img_sizeH, opt.img_sizeW)),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    transforms.Normalize((0.5,), (0.5,))
                    ]));
    else:           
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_path,
            transform=transforms.Compose([              
                    transforms.Resize((opt.img_sizeH, opt.img_sizeW)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]));
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader

def pause():
    print("Pause...")
    input()

def resize2d(img, size):
    return (F.adaptive_avg_pool2d(Variable(img,volatile=True), size)).data


#--------------------------------------------------------------------------MAIN
#Loss function
adversarial_loss = torch.nn.BCELoss()

#Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

#Shift to GPU if possible
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

#pause()

#load generator
generator.load_state_dict(torch.load(modelDir + modelFileName, map_location=lambda storage, loc: storage.cuda(0)))
generator.eval()

#loops
countImg = 1;
numLoopsGen = int(opt.n_images / opt.batch_size)
for b in range(numLoopsGen):
    #Sample noise as generator input
    z = Variable(Tensor(np.random.normal(0, 1, (opt.batch_size, opt.latent_dim))))

    #Generate a batch of images
    gen_imgs = generator(z)

    #print(gen_imgs.data.shape)

    #save
    for i in range(opt.batch_size):
        imgToSave = gen_imgs.data[i];
        imgToSave = imgToSave.view(opt.channels, opt.img_sizeH, opt.img_sizeW)
        #save_image(resize2d(imgToSave, (opt.img_sizeH, opt.img_sizeW)), outDir+'%d.jpg' % batches_done, nrow=1, normalize=True)
        save_image(imgToSave, outDir+'%d.jpg' % countImg, nrow=1, normalize=True)
        if (countImg % 100 == 0):
            print("Generating image n. %d / %d" % (countImg, opt.n_images))
        countImg = countImg + 1







            