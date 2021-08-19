# Graph Attention Multi-Layer Perceptron

## Environments

Implementing environment: Xeon(R) Platinum 8255C (CPU), 376GB(RAM), Tesla V100 32GB (GPU), Ubuntu 16.04 (OS).

## Requirements

Experimenting environments: Xeon(R) Platinum 8255C (CPU), 376GB(RAM), Tesla V100 32GB (GPU), Ubuntu 16.04 (OS)

The PyTorch version we use is torch 1.7.1+cu101. Please refer to the official website -- https://pytorch.org/get-started/locally/ -- for the detailed installation instructions.

To install other requirements:

```setup
pip install -r requirements.txt
```



## Training

To reproduce the results of **GAMLP+RDE** on OGB datasets, please run following commands.

For **ogbn-products**:

```bash
python main.py --use-rdd --method R_GAMLP_RDD --stages 400 300 300 300 --train-num-epochs 0 0 0 0 --threshold 0.85 --input-drop 0.2 --att-drop 0.5 --label-drop 0 --pre-process --residual --dataset ogbn-products --num-runs 10 --gpu 6 --eval 10 --act leaky_relu --batch 50000 --patience 300 --n-layers-1 4 --n-layers-2 4 --bns --gama 0.1
```

For **ogbn-papers100M**:

```bash
python main.py --use-rdd --method R_GAMLP_RDD --stages 100 150 150 150 --train-num-epochs 0 0 0 0 --threshold 0 --input-drop 0 --att-drop 0 --label-drop 0 --dropout 0.5 --pre-process --dataset ogbn-papers100M --num-runs 3 --eval 1 --act sigmoid --batch 5000 --patience 300 --n-layers-2 6 --label-num-hops 9 --num-hops 6 --hidden 1024 --bns --temp 0.001
```

For **ogbn-mag**:

```bash
python main.py --use-rdd --method JK_GAMLP_RDD --stages 250 200 200 200 --train-num-epochs 0 0 0 0 --threshold 0.4 --input-drop 0.1 --att-drop 0 --label-drop 0 --pre-process --residual --dataset ogbn-mag --num-runs 10 --eval 10 --act leaky_relu --batch 10000 --patience 300 --n-layers-1 4 --n-layers-2 4 --label-num-hops 3 --seed 0 --gpu 1 --bns --gama 10 --use-relation-subsets ./data/mag --emb_path ./data/
```



## Node Classification Results:

Performance comparison on **ogbn-products**:

![image-20210819193909175](./products_perf.png)

Performance comparison on **ogbn-papers100M**:

![image-20210819194124961](./papers100M_perf.png)

Performance comparison on **ogbn-mag**:

![image-20210819194235072](./mag_perf.png)
