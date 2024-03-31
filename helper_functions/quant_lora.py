import copy
import torch
from torch import nn
from train_evaluate.quant_func import *

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha, diff_matrix=None):
        super().__init__()
        if diff_matrix is not None:
            # decompose diff_matrix using SVD:
            U, S, V = torch.svd(diff_matrix)
            # reconstruct lower rank matrix:
            self.A = nn.Parameter(U[:, :rank] @ torch.diag(torch.sqrt(S[:rank])))
            self.B = nn.Parameter(torch.diag(torch.sqrt(S[:rank])) @ V[:, :rank].T)
            alpha = 1
        else:
            std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
            self.A = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
            self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        # x @ A = in_dimx1 @ in_dimxrank -> in_dim*rank
        # (x @ A) @ B = 1xrank @ rankxout_dim -> rank*out_dim
        x = self.alpha * (x @ (self.A @ self.B).T)
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha, diff_matrix=None):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha, diff_matrix=diff_matrix
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

def ReplaceLinearToLoRA(model, rank, alpha, diff_matrices=None):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LinearWithLoRA(module, rank=rank, alpha=alpha, diff_matrix=diff_matrices[name]))
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
    diff_matrices = {}

    # Copy the model to avoid modifying the original model
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Extract weights and biases
            weight = module.weight
            # bias = module.bias

            # deep copy the original weight matrix:
            diff_matrices[name] = copy.deepcopy(weight)

            # Quantize weights
            if quant_type == "sparse":
                quantized_weight = quant_tensor_sparse(weight, size)
            elif quant_type == "mean":
                quantized_weight = quant_tensor_mean(weight, size)
            elif quant_type == "int8":
                quantized_weight = (weight * 127).round().clamp(-128, 127).to(torch.int8) / 127
            elif quant_type == "int1":
                quantized_weight = weight.sign()
            else:
                quantized_weight = weight
            module.weight = nn.Parameter(quantized_weight, requires_grad=False)
            diff_matrices[name] = diff_matrices[name] - quantized_weight

            # # Optional: Quantize biases if they exist
            # if quant_type == "sparse":
            #     quantized_bias = quant_tensor_sparse(bias, size)
            # elif quant_type == "mean":
            #     quantized_bias = quant_tensor_mean(bias, size)
            # elif quant_type == "int8":
            #     quantized_bias = (bias * 127).round().clamp(-128, 127).to(torch.int8) / 127
            # elif quant_type == "int1":
            #     quantized_bias = bias.sign()
            # else:
            #     quantized_bias = bias
            # module.bias = nn.Parameter(quantized_bias, requires_grad=False)

    return diff_matrices

def quantize_lora(model, lora_rank, alpha, quant_type=None, quant_size=4*4, is_svd=True):
    # Quantize the model to the specified dtype and replace the linear layers with LoRa layers
    diff_matrices = quantize_linear_layers(model, quant_type, quant_size)
    if is_svd:
        ReplaceLinearToLoRA(model, lora_rank, alpha, diff_matrices)
    else:
        ReplaceLinearToLoRA(model, lora_rank, alpha)
    FreeazeModel(model)
    UnfreezeLoRA(model)
