import torch
from train_evaluate.quant_func import *

def quant_tensor_sparse(tensor, size=4*4):
    quant_model_flatten = tensor
    if tensor.numel() < size:
        size = tensor.numel()

    orig_shape = quant_model_flatten.shape
    quant_model_flatten = quant_model_flatten.flatten()

    with torch.no_grad():
        for i in range(0, len(quant_model_flatten), size):
            # Get the maximum value in the subsection:
            max_val = quant_model_flatten[i:i+size].max()
            max_index = quant_model_flatten[i:i+size].argmax()
            # Set the subsection to zero:
            quant_model_flatten[i:i+size] = torch.zeros_like(quant_model_flatten[i:i+size])
            # Restore only 'max_val' to 'max_index' in the subsection:
            quant_model_flatten[i+max_index] = max_val
        quant_model_flatten = quant_model_flatten.view(orig_shape)

    return quant_model_flatten

def quant_tensor_mean(tensor, size=4*4):
    quant_model_flatten = tensor
    if tensor.numel() >= size:
        orig_shape = quant_model_flatten.shape
        quant_model_flatten = quant_model_flatten.flatten()
        quant_model_flatten = quant_model_flatten.unfold(0, size, size).mean(-1)
        quant_model_flatten = quant_model_flatten.repeat_interleave(size)
        quant_model_flatten = quant_model_flatten.view(orig_shape)
    else:
        quant_model_flatten = quant_model_flatten.mean() * torch.ones_like(quant_model_flatten)

    return quant_model_flatten