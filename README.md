# GIANT-XRT with GAMLP+RLU

This is the repository for reproducing the results in our paper: [[Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction]](https://arxiv.org/pdf/2111.00064.pdf) for the combination of GIANT-XRT with GAMLP+RLU.

## Step 0: Install GIANT and get GIANT-XRT node features.
Please follow the instruction in [[GIANT]](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt) to get the GIANT-XRT node features.

## Step 1: Git clone this repo.
After following the steps in [[GIANT]](https://github.com/amzn/pecos/tree/mainline/examples/giant-xrt), go to the folder
`pecos/examples/giant-xrt`
Then git clone this repo in the folder `giant-xrt` directly.

## Step 2: Install additional packages.
If you install and run GIANT correctly, you should only need to additionally install [dgl>=0.6.1](https://github.com/dmlc/dgl).
See [here](https://www.dgl.ai/pages/start.html) for pip/conda installation instruction for dgl.
In our experiment, we use the following
```
# DGL's cudnn version need to be exactly the same as locally-installed CUDA version!
# for our p3dn.24xlarge, its cu11.0 now
pip install dgl-cu110 -f https://data.dgl.ai/wheels/repo.html
```

## Step 3: Run the experiment.
Go to the folder `giant-xrt`. 

To reproduce the GIANT-XRT results of **GAMLP+RLU** on OGB datasets, please run following commands.

For **ogbn-products**:
```bash
GPU_ID=0
NODE_EMB_PATH=../proc_data_xrt/ogbn-products/X.all.xrt-emb.npy
bash run_gamlp_xrt.sh ogbn-products ${NODE_EMB_PATH} ${GPU_ID}
```

For **ogbn-papers100M**:
```bash
GPU_ID=0
NODE_EMB_PATH=../proc_data_xrt/ogbn-papers100M/X.all.xrt-emb.npy
bash run_gamlp_xrt.sh ogbn-papers100M ${NODE_EMB_PATH} ${GPU_ID}
```

## Results
If execute correctly, you should have the following performance (using our pretrained GIANT-XRT features).

**GAMLP+RLU** Number of params: 21,551,631

| GAMLP+RLU | stage 0 | stage 1 | stage 2 | stage 3
|---|---|---|---|
| Average val accuracy (%) | 72.61±0.04 | 72.89±0.02 | 72.99±0.03 | 73.05±0.04
| Average test accuracy (%) | 69.16±0.08 | 69.50±0.08 | 69.61±0.08 | 69.67±0.05

**Remark:** We follow default hyper-parameters of GAMLP+RLU for our GIANT-XRT. It is possible to achieve higher performance by fine-tune it more carefully.

For more details about GAMLP, please check their original [README](https://github.com/PKU-DAIR/GAMLP).

## Citation
If you find our code useful, please consider citing both our paper and GAMLP work.

Our GIANT-XRT paper:
```
@article{chien2021node,
  title={Node Feature Extraction by Self-Supervised Multi-scale Neighborhood Prediction},
  author={Eli Chien and Wei-Cheng Chang and Cho-Jui Hsieh and Hsiang-Fu Yu and Jiong Zhang and Olgica Milenkovic and Inderjit S Dhillon},
  journal={arXiv preprint arXiv:2111.00064},
  year={2021}
}
```

GAMLP paper:
```
@article{zhang2021graph,
  title={Graph Attention Multi-Layer Perceptron},
  author={Zhang, Wentao and Yin, Ziqi and Sheng, Zeang and Ouyang, Wen and Li, Xiaosen and Tao, Yangyu and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2108.10097},
  year={2021}
}
```
i
