import torch
import torchvision
from torch import nn

######pre-trained
vgg16 = torchvision.models.vgg16(pretrained=True) 
#vgg16.classifier.add_module('add_linear', nn.Linear(1000, 10))
vgg16.classifier[6] = nn.Linear(4096, 10)
#print(vgg16)
model_save_path='./cifar_net.pth'

#####save
##method_1: save parameter
#torch.save(vgg16.state_dict(), "vgg16_method_1.pth")
##method_2: save model and parameter
#vgg16 = torchvision.models.vgg16(pretrained=False)
torch.save(vgg16, model_save_path)

#####load
#method_1: save parameter
#vgg16 = torchvision.models.vgg16(pretrained=False)
#vgg16.load_state_dict(torch.load("vgg16_method_1.pth"))
##method_2: save model and parameter
#model = torch.load("vgg16_method_2.pth")
