import torch
import torchvision
import timm

def get_model():
    # Load the pretrained model
    model = timm.create_model("resnet18_cifar10", pretrained=True)
    #model = timm.create_model("resnet18_cifar100", pretrained=True)
    #model = torchvision.models.resnet18(pretrained=True)
    return model