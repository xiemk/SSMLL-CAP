
import torch
from torchvision import models

def create_model(mo, n_classes):

    model = resnet(mo.lower(), n_classes)

    return model

def resnet(mo, nc, pretrain=True):
    
    if mo == 'resnet18':
        model = models.resnet18(pretrained=pretrain)
    elif mo == 'resnet32':
        model = models.resnet32(pretrained=pretrain)
    elif mo == 'resnet50':
        model = models.resnet50(pretrained=pretrain)
    
    if mo in ['resnet18','resnet32', 'resnet50']:
        model.fc = torch.nn.Linear(model.fc.in_features, nc)
        
    return model
