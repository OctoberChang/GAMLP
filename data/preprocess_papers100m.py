import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, dropout_adj
import os
import os.path as osp

from ogb.nodeproppred import PygNodePropPredDataset


parser = argparse.ArgumentParser()
parser.add_argument('--num_hops', type=int, default=6)
parser.add_argument('--root', type=str, default='./')
parser.add_argument('--pretrained_emb_path', type=str, default=None)
parser.add_argument('--output_emb_prefix', type=str, default='./ogbn-papers100M_node-emb_w2v')
args = parser.parse_args()
print(args)

dataset = PygNodePropPredDataset('ogbn-papers100M', root=args.root)
split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
data = dataset[0]

x = None
if args.pretrained_emb_path is not None:
    x = np.load(args.pretrained_emb_path)
else:
    x = data.x.numpy()
N = data.num_nodes
print("Node Embeddings", x.shape)

path = './adj_gcn.pt'
if osp.exists(path):
    adj = torch.load(path)
else:
    print('Making the graph undirected.')
    # Randomly drop some edges to save computation
    data.edge_index = to_undirected(data.edge_index, num_nodes=data.num_nodes)

    print(data)

    row, col = data.edge_index

    print('Computing adj...')

    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    adj = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    torch.save(adj, path)


adj = adj.to_scipy(layout='csr')

print('Start processing')

saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_0.pt')

for i in tqdm(range(args.num_hops)):
    x = adj @ x
    saved = np.concatenate((x[train_idx], x[valid_idx], x[test_idx]), axis=0)
    torch.save(torch.from_numpy(saved).to(torch.float), f'{args.output_emb_prefix}_{i+1}.pt')
