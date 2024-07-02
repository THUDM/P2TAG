## Datasets

We utilize two categories of datasets in our project: **OGBN** and **Amazon Review**.

### Amazon Review Dataset

The **Amazon Review** dataset is available from this [link](https://pan.baidu.com/s/1jQVaC80H4rx1wCXkBZ146w?pwd=rdnq). This dataset is compiled based on [Yan's research work](https://proceedings.neurips.cc/paper_files/paper/2023/file/37d00f567a18b478065f1a91b95622a0-Paper-Datasets_and_Benchmarks.pdf). We re-split it for few-shot node classification. The label files are located under the path `split/class_split.json` for each dataset.

### OGBN Dataset

The **OGBN** dataset is set to be downloaded automatically. However, users are required to follow the instructions provided on the [OGBN official website](https://ogb.stanford.edu/docs/nodeprop/) to obtain the raw text for the 'ogbn-arxiv' and 'ogbn-products' datasets.  We also provide the corresponding Baidu Cloud [link](https://pan.baidu.com/s/1hM38L5d73i1UBtAYBUgtpg?pwd=6zt3).

To ensure the reproducibility of our results, please copy the corresponding files from the 'preprocess/' folder into the 'ogbn_arxiv' and 'ogbn_products' folders. This step is crucial as it guarantees the same data split is used.