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
#imgDir = directory where are the rubber sheet models
imgDir = './rsm/'
outDir = './images/'

#Create directory
os.makedirs(outDir, exist_ok=True)

#Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=60, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_sizeH', type=int, default=64, help='height size of each image dimension')
parser.add_argument('--img_sizeW', type=int, default=512, help='width size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--model_save_interval', type=int, default=10000, help='interval between model saving')
opt = parser.parse_args()
print(opt)


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
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128*self.init_sizeH*self.init_sizeW)) #L1

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), #L2
            nn.Upsample(scale_factor=2), #L2
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #L3
            nn.BatchNorm2d(128, 0.8), #L3
            nn.LeakyReLU(0.2, inplace=True), #L4
            nn.Upsample(scale_factor=2), #L4
            nn.Conv2d(128, 64, 3, stride=1, padding=1), #L5
            nn.BatchNorm2d(64, 0.8), #L5
            nn.LeakyReLU(0.2, inplace=True), #L6
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_sizeH, self.init_sizeW) #L2
        img = self.conv_blocks(out)

        #print("img uscita da forward: ")
        #print(out.shape)

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
            *discriminator_block(opt.channels, 16, bn=False), #block 1 L1-3
            *discriminator_block(16, 32), #block 2 L4-6
            *discriminator_block(32, 64), #block 3 L7-9
            *discriminator_block(64, 128), #block 4 L10-12
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

#Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)
        
#Load data
dataloader = load_dataset();    

#pause()

#Initialize optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))


# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        #print(imgs.shape)

        #Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        #Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        #Generate a batch of images
        gen_imgs = generator(z)

        #print(gen_imgs.shape)

        #Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        #Training step
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        #Configure input
        real_imgs = Variable(imgs.type(Tensor))
       
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        #Training step
        d_loss.backward()
        optimizer_D.step()
        
        #Display
        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        

        #Save output
        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            imgToSave = gen_imgs.data[:1];
            imgToSave = imgToSave.view(opt.channels, opt.img_sizeH, opt.img_sizeW)
            #save_image(resize2d(imgToSave, (opt.img_sizeH, opt.img_sizeW)), outDir+'%d.jpg' % batches_done, nrow=1, normalize=True)
            save_image(imgToSave, outDir+'%d.jpg' % batches_done, nrow=1, normalize=True)
            
        if batches_done % opt.model_save_interval == 0:
            filenameSave = "./models/model_save_%d.pth" % batches_done 
            with open(filenameSave, 'wb') as f: 
                #torch.save(generator, f)
                torch.save(generator.cpu().state_dict(), f)
                generator.cuda()
    
        
        
        
        
            