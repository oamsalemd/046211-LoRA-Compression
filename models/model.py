import torch
import torchvision
import timm
import detectors

def get_cifar10_model():
    # Load the pretrained model
    model = timm.create_model("resnet18_cifar10", pretrained=True)
    #model = timm.create_model("resnet18_cifar100", pretrained=True)
    #model = torchvision.models.resnet18(pretrained=True)
    return model

def get_imagenet_model():
    # Load the pretrained model
    model = timm.create_model("resnet18.tv_in1k", pretrained=True)
    return model