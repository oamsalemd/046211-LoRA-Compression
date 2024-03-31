# 046211-LoRA-Compression
@oamsalemd, @idob8 - Winter 2024

# Ethics Statement
### Stakeholders
End-users, deep learning researchers, technology companies, and regulatory bodies.
### Implications
End-users can benefit from faster and more efficient image classification models, improving user experience. However, there may be concerns about privacy if sensitive information is processed.
Deep learning researchers can advance the field with innovative techniques, but they must ensure fairness and transparency in model development and deployment.
Technology companies can enhance product performance and reduce resource consumption, yet they need to address potential biases and ensure responsible AI practices.
Regulatory bodies play a crucial role in establishing guidelines and standards to protect user rights, promote fairness, and mitigate risks associated with AI technologies.
### Ethical Considerations
Prioritizing user privacy and data protection through robust security measures and transparent data handling practices.
Mitigating biases in data and algorithms to ensure fairness and equity in classification outcomes.
Providing clear explanations and documentation on the use of quantization and LoRA techniques to enhance model transparency and interpretability.
Engaging in ongoing dialogue with stakeholders and regulatory bodies to address ethical concerns, promote responsible AI practices, and uphold societal values in AI development and deployment.

# How to run
## Environments settings
## Execution commands

# Project documentation
## Topics
* Introduction
* Compression
* LoRA
* Method
* Experiments and results
* Conclusions and future work

## Introduction
Compressing pre-trained neural networks reduces memory usage, speeds up inference, and enables deployment on resource-constrained devices. It optimizes model efficiency, real-time performance, and energy consumption, making deep learning models more practical for diverse computing environments.
![ProjectPresentation](https://github.com/oamsalemd/046211-LoRA-Quantization/assets/93587192/b4584862-d78d-4784-b2a3-4e8117ca3338)

LoRA (Low Rank Adaptation) is a technique for efficiently fine-tune pre-trained models. The basic idea is to train only a low-rank matrix that will be added the pretrained weight matrix.<sup>[1]</sup>

![image](https://github.com/oamsalemd/046211-LoRA-Quantization/assets/93587192/6a492711-a3e1-4a4c-8188-b746ff88c304)

Our objective is to combine model compression with LoRA in pre-trained models, to optimize model size with minimal demaging to model accuracy and minimal retraining. Previous works have shown the benefits of LoRA in transfer-learning for pre-trained LLM-s.<sup>[1]</sup> We test the method's efficacy for image classification tasks.

## Compression
We tested multiple model compression methods that can potentially achieve better computational usage, and tested their effect on the pre-trained model.
Data type Quantization - in this method we use more compact data type to store the model weights. this technique can potentially save memory (capacity and bandwidth). 
Sparsity - in this method we use "sparse" weight matrices, for any given block we allow only 1 cell to have non zero value. This technique can potentially save memory (capacity and bandwidth) and also reduce the number of effective multiplication instruction. 
On the other hand, both methods can potentially damage the accuracy of the model and might demand retraining the model.

## LoRA
- Given a 'Linear' layer `W` of `in_dimXout_dim`, we choose low rank `r` s.t. `r < in_dim, out_dim`.
- We freeze the `W` matrix, so it remains intact while re-training the model.
- The matrices `A` (of `in_dimXr`) and `B` (of `rXout_dim`) and initialized.
- We set the new activation to be `h=x@(W+a*A@B)` for the input `x` (of `1Xin_dim`), and a factor `a`.
- During training, only `A` and `B` matrices are learned.

Benefit: reduce computation resources while training

## Method
- Create pre-trained model for image classification
  - We used ‘resnet18’ pre-trained on ImageNet1K<sup>[2]</sup>
- Quantize the ‘Linear’ layer(s) in the model
  - We tested numerous quantization methods, as described above
- Freeze the model parameters
- Add trainable LoRA to quantized ‘Linear’ layer(s)
- Train the model with the same pre-trained dataset
  - Using Optuna for best training hyper-parameters
- Evaluate accuracy

## Experiments and results
(-> summary graph from 'Accuracy' datasheet)
(-> pointer to 'results' directory with output plots)

## Conclusions and future work




> <sup>[1]</sup> Hu, Edward J., et al. “Lora: Low-rank adaptation of large language models.” arXiv preprint arXiv:2106.09685 (2021).

> <sup>[2]</sup> https://huggingface.co/timm/resnet18.tv_in1k
