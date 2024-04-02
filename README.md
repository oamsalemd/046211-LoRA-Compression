# 046211-LoRA-Compression
@oamsalemd, @idob8 - Winter 2024


# Project documentation
## Topics
* Introduction
  * Compression
  * LoRA
* Project goal
* Method
* Experiments and results
* Conclusions and future work

## Introduction
## Compression
Compressing pre-trained neural networks reduces memory usage, speeds up inference, and enables deployment on resource-constrained devices. It optimizes model efficiency, real-time performance, and energy consumption, making deep learning models more practical for diverse computing environments.
We tested multiple model compression methods that can potentially achieve better computational usage, and tested their effect on the pre-trained model.

Data type Quantization - in this method we use more compact data type to store the model weights. this technique can potentially save memory (capacity and bandwidth).

Sparsity - in this method we use "sparse" weight matrices, for any given block we allow only 1 cell to have non zero value. This technique can potentially save memory (capacity and bandwidth) and also reduce the number of effective multiplication instruction.

On the other hand, both methods can potentially damage the accuracy of the model and might demand retraining the model.
![ProjectPresentation](https://github.com/oamsalemd/046211-LoRA-Quantization/assets/93587192/b4584862-d78d-4784-b2a3-4e8117ca3338)


## LoRA
LoRA (Low Rank Adaptation) is a technique for efficiently fine-tune pre-trained models. The basic idea is to train only a low-rank matrix that will be added the pretrained weight matrix.<sup>[1]</sup>
Previous works have shown the benefits of LoRA in transfer-learning for pre-trained LLM-s.<sup>[1]</sup>
- Given a 'Linear' layer `W` of `in_dimXout_dim`, we choose low rank `r` s.t. `r < in_dim, out_dim`.
- We freeze the `W` matrix, so it remains intact while re-training the model.
- The matrices `A` (of `in_dimXr`) and `B` (of `rXout_dim`) and initialized.
- We set the new activation to be `h=x@(W+a*A@B)` for the input `x` (of `1Xin_dim`), and a factor `a`.
- During training, only `A` and `B` matrices are learned.

![image](https://github.com/oamsalemd/046211-LoRA-Quantization/assets/93587192/6a492711-a3e1-4a4c-8188-b746ff88c304)

## Project goal
Our objective is to combine model compression with LoRA in pre-trained models, to optimize model size with minimal demaging to model accuracy and minimal retraining.  We test the method's efficacy for image classification tasks.

## Method
- We used ‘resnet18’ pre-trained on ImageNet1K<sup>[2]</sup>
  - For training we used only a small subset of the original dataset (50,000 images out of 1,281,167)
- The compression methods we tested were:
  - Data type quantization to int1.
  - Sparsity with block size of 4X4.
- The compression was implemented only on the FC layer of the model.
- We tested the appending of LoRA layer of ranks: [2, 4, 8, 16, 32, 64, 128].
  - We tested 2 initialization methods. The first was the initialization suggested in the original LoRA paper, A initialize as N(0,\sigma^2) and B=0. The second one was SVD decomposition of the diff from original matrix.
- All model's parameters except LoRA parameters was freezed. LoRA parameters were trained for 10 epochs and the best epoch was chosen (in terms of accuracy on the validation set).
- Hyper parameters were chosen for each rank seperatly using OPTUNA
  - Optimizer, learning rate, batch size, "alpha" (LoRA)
- Finally we evaluated the accuracy on a test set for each LoRA rank and for each initialization method.

## Experiments and results
(-> summary graph from 'Accuracy' datasheet)
(-> pointer to 'results' directory with output plots)

## Conclusions
- Increasing LoRA rank gives better accuracy, as expected, yet not matching the original model’s accuracy.
- Training the LoRA parameters requires minor computation effort.
- The combnation of all LoRA ranks with compression methods that were tested results in memory compression, sparsity method also results in computation compression.
- LoRA training is unstable and very prone to hyper-parameters modification.
- Using initialization with SVD decomposition results in...

## Future wotk
We believe that our project shows potintial for further research of the benefits from combining model compression methods with LoRA.
We believe such research could be done with:
- Test the method’s performance for ‘Linear’-rich models (e.g. Transformers, MLP-based, …)
- Explore more compression hyper-parameters (e.g. int8, sparse 3X3, ...)
- Explore more initialization methods for the LoRA matrices



# How to run

## Environments settings
1. Clone to a new directory:
  `git clone <URL> <DEST_DIR>`
2. `cd /path/to/DEST_DIR`
3. `pip install -r requirements.txt`
4. Download ImageNet subset from:
  https://www.kaggle.com/datasets/tusonggao/imagenet-validation-dataset/code
5. Move the images directory to:
  `DEST_DIR/../archive/imagenet_validation`

## Execution commands

`python train_evaluate/train_model.py --init {paper_init,svd_init} [ > log.txt]`
* `--init`: determines the LoRA matrices initialization method (default: `paper_init`)
* Recommended: pipe the output to `log.txt` file
* Results will appear in `DEST_DIR/results`

Description:
1. Initiates the `resnet18` model, pretrained on ImageNet
2. Per each compression method (`sparse`, `int1`):
  * Compresses the model
  * Initiates LoRA appended to FC layer(s)
  * Sweeps LoRA rank values, and uses Optuna to find the best training hyper-parameters per each rank
  * Outputs the results to a dedicated directory
3. Results directory contains:
  * `evaluation.csv`: summary of evaluation accuracy for test subset per each LoRA rank
  * `acc_quant=COMP_TYPE_r=RANK.png`: accuracy per epoch (train, validation), for COMP_TYPE (`sparse`, `int1`), for RANK (LoRA rank)
  * `loss_quant=COMP_TYPE=r_RANK.png`: loss per epoch (train, validation), for COMP_TYPE (`sparse`, `int1`), for RANK (LoRA rank)
  * `quant=COMP_TYPE_r=RANK_eval_acc=ACCUR.ckpt`: model post-training parameters, for COMP_TYPE (`sparse`, `int1`), for RANK (LoRA rank), with test accuracy of ACCUR
  * `quant=COMP_TYPE_r=RANK_optimization_history.html`: Optuna trials summary for COMP_TYPE (`sparse`, `int1`) and RANK (LoRA rank) hyper-parameter tuning


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




> <sup>[1]</sup> Hu, Edward J., et al. “Lora: Low-rank adaptation of large language models.” arXiv preprint arXiv:2106.09685 (2021).

> <sup>[2]</sup> https://huggingface.co/timm/resnet18.tv_in1k
