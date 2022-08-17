import os,sys

from random_word import RandomWords
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import requests
random_word_generator = RandomWords()

class ResNetAttributes(nn.Module):
    def __init__(self, 
                save_path= "./models",
                model_path="./models",
                pretrained=False,
                num_classes=40,
                version='resnet18',
                name=None,
                device='cuda:0' if torch.cuda.is_available() else 'cpu',
                ):
        super(ResNetAttributes, self).__init__()
        self.save_path = save_path
        if name is None:
            try:
                name = f"{random_word_generator.get_random_word()}_{random_word_generator.get_random_word()}"
            except:
                name = "NoCennection"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.pretrained = pretrained
        self.model_path = model_path
        if not os.path.exists(model_path):
            raise Exception(f"{model_path} not found")
        self.num_classes = num_classes
        self.version = version
        self.name = name
        self.device = device
        if version not in ['resnet18','resnet34','resnet50','resnet101','resnet152']:
            version = 'resnet18'
        if not pretrained:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', f'{version}', pretrained=True)
            self.model.fc = nn.Linear(512, num_classes)
            self.model = nn.Sequential(self.model, nn.Sigmoid())
        else:
            self.model = torch.hub.load('pytorch/vision:v0.10.0', f'{version}', pretrained=True)
            self.model.fc = nn.Linear(512, num_classes)
            self.model = nn.Sequential(self.model, nn.Sigmoid())
            print(f"Loading pretrained model from {model_path}")
            self.model.load_state_dict(torch.load(self.model_path))
        self.to(self.device)
    def save(self, path=None,best=False,epoch=0):
        if path is None:
            path = self.save_path
        if best:
            save_path = path+'/'+self.name+'_best.pth'
        else:
            save_path = path+'/'+self.name+'_'+str(epoch)+'.pth'

        torch.save(self.state_dict(), save_path)
        print(f"model saved to {path}")
    def forward(self, x):
        return self.model(x)
    def get_embedding(self, x):
        return self.model[:-2](x)

if __name__=="__main__":
    model = ResNetAttributes(save_path="./models")
    random_input = torch.randn(10,3,224,224)
    print(f"Input Shape: {random_input.shape}")
    output = model(random_input)
    print(f"Output Shape: {output.shape}")
    model.save()

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()