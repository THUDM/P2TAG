## Official Implementation of the P2TAG Paper

This code repository is the official implementation of the P2TAG paper.

## Requirements

<!-- ```plaintext -->
Most of the requirements are listed in the requirements.txt file.
The torch-geometric and dgl libraries are required. The specific versions depend on your CUDA driver version.
Please refer to their official websites for installation instructions.
- torch==1.10.2
- torch-geometric==2.0.3
- dgl==0.8.0
<!-- ``` -->

## Datasets

As shown in the paper, we utilize the following datasets: `["arxiv", "products", "computers", "children", "history", "photo"]`. We provide the datasets for "computers", "children", "history", and "photo" in the `./data` folder. 

The ‘arxiv’ and ‘products’ datasets can be downloaded automatically the first time the code is run. The raw text for the nodes in these two datasets should be downloaded according to the instructions on the [OGB website](https://ogb.stanford.edu/docs/nodeprop/). For a detailed introduction, please refer to this [document](./dataset/README.md).


## Running the Experiments

### Pre-training

Pre-training experiments can be run using the `pre_train.sh` script located in the `./bash_scripts` folder. We use WandB ([https://wandb.ai/](https://wandb.ai/)) for experiment management. Please specify `run_entity` as your WandB username in the running scripts.

### Few-shot Node Classification

Experiments for few-shot node classification can be run using the `few_shot_eval.sh` script in the `./bash_scripts` folder.

### Introduction to Hyper-parameters

Hyper-parameters are specified in `utils/functions.py`. Important hyper-parameters include:

```python
- label_as_init: True/False denotes the P_{G} component in the P2TAG model.
- LM_as_init: True/False denotes the W_{t} component in the P2TAG model.
- prompt_type: "default"/"only_first" (readout type, where default is mean pooling, and only_first uses the first node embedding).
- few_shot_setting: "3,3"/"3,5"/"3,10" (denotes the few-shot setting, 3-way 3-shot, 3-way 5-shot, 3-way 10-shot, respectively).
```

### Examples of Running Pre-training and Few-shot Node Classification

```plaintext
# Pre-training
bash bash_scripts/pre_train.sh

# Few-shot node classification
bash bash_scripts/few_shot_eval.sh
```

### Reproduce scripts

We provide the scripts to reproduce the results in the `./reproduce_scripts` folder, one can run the scripts to reproduce the results in the paper. We also provide pre-trained models. If your focus is only on the prompting process, you can download the models from this [link](https://pan.baidu.com/s/1tsUII6mP1D9P9hVpN-Ar-w?pwd=yoel).
