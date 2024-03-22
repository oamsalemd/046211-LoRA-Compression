import torch
from torch import nn
from train_evaluate.quant_func import *

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        # x @ A = in_dimx1 @ in_dimxrank -> in_dim*rank
        # (x @ A) @ B = in_dimxrank @ rankxout_dim -> in_dim*rank*out_dim
        x = self.alpha * (x @ self.A @ self.B)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def ReplaceLinearToLoRA(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank=rank, alpha=alpha))
        else:
            ReplaceLinearToLoRA(module, rank=rank, alpha=alpha)

def FreeazeModel(model):
    for param in model.parameters():
        param.requires_grad = False

def UnfreezeLoRA(model):
    for child in model.children():
        if isinstance(child, LoRALayer):
            for param in child.parameters():
                param.requires_grad = True
        else:
            # Recursively freeze linear layers in children modules
            UnfreezeLoRA(child)

def quantize_linear_layers(model, quant_type=None, size=4*4):
    # Copy the model to avoid modifying the original model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract weights and biases
            weight = module.weight
            bias = module.bias

            # Quantize weights
            if quant_type == "sparse":
                quantized_weight = quant_tensor_sparse(weight, size)
            elif quant_type == "mean":
                quantized_weight = quant_tensor_mean(weight, size)
            else:
                quantized_weight = weight
            module.weight = nn.Parameter(quantized_weight, requires_grad=False)

            # Optional: Quantize biases if they exist
            if quant_type == "sparse":
                quantized_bias = quant_tensor_sparse(bias, size)
            elif quant_type == "mean":
                quantized_bias = quant_tensor_mean(bias, size)
            else:
                quantized_bias = bias
            module.bias = nn.Parameter(quantized_bias, requires_grad=False)

def quantize_lora(model, lora_rank, alpha, quant_type=None, quant_size=4*4):
    # Quantize the model to the specified dtype and replace the linear layers with LoRa layers
    quantize_linear_layers(model, quant_type, quant_size)
    ReplaceLinearToLoRA(model, lora_rank, alpha)
    FreeazeModel(model)
    UnfreezeLoRA(model)
