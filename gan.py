#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 13:32:18 2017

@author: loganrooks
"""
from __future__ import print_function
import torch
import torch.nn as nn
import nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils 
from torch.autograd import Variable




        
class Generator(nn.module):
    
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
                *[*self._deconvolutionBlock(inChannels, outChannels) for 
                  (inChannels, outChannels) in ((512, 256), (256, 128), (128, 64), (64, 3))],
                nn.Tanh()
                )
        
    def _deconvolutionBlock(self, inChannels, outChannels):
        block = [nn.BatchNorm2d(inChannels),
                 nn.ReLU(True),
                 nn.ConvTranspose2d(inChannels, outChannels, 4, 2, 1, bias = False)]
        return block
    
    def forward(self, inputImage):
        output = self.main(inputImage)
        return output
    
class Discriminator(nn.module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
                *[*self._deconvolutionBlock(inChannels, outChannels) for 
                  (inChannels, outChannels) in ((3, 64), (64, 128), (128, 256), (256, 512))],
                  nn.Conv2d(512, 1, 4, 1, 0, bias = False),
                  nn.Sigmoid()
                )
        
    def _convolutionBlock(self, inChannels, outChannels):
        block = [nn.Conv2d(inChannels, outChannels, 4, 2, 1, bias = False),
                 nn.BatchNorm2d(outChannels),
                 nn.LeakyReLU(0.2, inplace = True)]
        return block
    
    def forward(self, inputImage):
        output = self.main(inputImage)
        return output.view(-1)
    
class GenerativeAdversarialNet():
    
    def __init__(self, learningRate, betas):
        self.discriminator = Discriminator().apply(self.weightsInit)
        self.generator = Generator().apply(self.weightsInit)
        self.discriminatorOptimizer = optim.Adam(self.discriminator.parameters(), learningRate, betas)
        self.generatorOptimizer = optim.Adan(self.generator.paramters(), learningRate, betas)
    
    def weightsInit(neuralNet):
        classname = neuralNet.__class__.__name__
        if classname.find('Conv') != -1:
            neuralNet.weightdata.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            neuralNet.weight.data.normal_(1.0, 0.02)
            neuralNet.bias.data.fill_(0)
            
    def train(numEpochs, batchSize, dataset, lossFunction):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchSize, shuffle = True, num_workers = 2)
        
        for epoch in range(numEpochs):
            
            for i, data in enumerate(dataloader, 0):
                
                discriminator.zero_grad()
                realImage, _ = data
                inputImage = Variable(realImage)
                target = Variable(torch.ones(inputImage.size()[0]))
                output = self.discriminator(inputImage)
                realImageErrorD = lossFunction(output, target)
                
                noise = Variable(torch.randn(inputImage.size()[0], 100, 1, 1))
                fakeImage = generator(noise)
                target = Variable(torch.zeros(inputImage.size()[0]))
                output = discriminator(fakeImage.detach())
                fakeImageErrorD = lossFunction(output, target)
                
                errorD = realImageError + fakeImageError
                error.backward()
                self.discriminatorOptimizer.step()
                
                self.generator.zero_grad()
                target = Variable(torch.ones(inputImage.size()[0]))
                output = self.discriminator(fakeImage)
                
                errorG = lossFunction(output, target)
                errorG.backward()
                self.generatorOptimizer.step()
                
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
                
                if i % 100 == 0:
                    vutils.save_image(realImage, '%s/real_samples.png' % "./results", normalize = True)
                    fakeImage = generator(noise)
                    vutils.save_image(fakeImage.data, '%s/fake_samples_epoch_%03d.png' % ("./results", epoch), normalize = True)
                
                
                
if __name__ == "__main__":
    imageSize = 64

    transform = transforms.Compose([transforms.Scale(imageSize), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    dataset = dset.CIFAR10(root = './data', download = True, transform = transform)
        