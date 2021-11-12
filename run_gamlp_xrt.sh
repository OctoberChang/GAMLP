
DATA_ROOT_DIR=../dataset

dataset=$1
input_emb_path=$2
gpu=$3
if [ -z ${dataset} ] || [ -z ${input_emb_path} ] || [ -z ${gpu} ]; then
    echo "USAGE: bash run_gamlp_xrt.sh [dataset] [input_emb_path] [gpu]"
    exit
fi

if [ ${dataset} == "ogbn-products" ]; then
    NUM_RUNS=10
    python main.py \
        --gpu ${gpu} \
        --num-runs ${NUM_RUNS} \
        --root ${DATA_ROOT_DIR} \
        --dataset ${dataset} \
        --node_emb_path ${input_emb_path} \
        --use-rlu \
        --method R_GAMLP_RLU \
        --stages 400 300 300 300 \
        --train-num-epochs 0 0 0 0 \
        --threshold 0.85 \
        --input-drop 0.2 \
        --att-drop 0.5 \
        --label-drop 0 \
        --pre-process \
        --residual \
        --eval 10 \
        --act leaky_relu \
        --batch 50000 \
        --patience 300 \
        --n-layers-1 4 \
        --n-layers-2 4 \
        --bns --gama 0.1 \
        |& tee ./gamlp.${dataset}.giant-xrt.log
    ##
elif [ ${dataset} == "ogbn-papers100M" ]; then
    NUM_RUNS=3
    num_hops=6
    output_emb_prefix=./ogbn-papers100M.node-emb.giant-xrt
    if [ ! -f "${output_emb_prefix}_${num_hops}.pt" ]; then
        python -u ./data/preprocess_papers100m.py \
            --root ${DATA_ROOT_DIR} \
            --num_hops ${num_hops} \
            --pretrained_emb_path ${input_emb_path} \
            --output_emb_prefix ${output_emb_prefix}
    fi
    python -u main.py \
        --gpu ${gpu} \
        --num-runs ${NUM_RUNS} \
        --root ${DATA_ROOT_DIR} \
        --dataset ${dataset} \
        --node_emb_path ${output_emb_prefix} \
        --use-rlu \
        --method R_GAMLP_RLU \
        --stages 100 150 150 150 \
        --train-num-epochs 0 0 0 0 \
        --threshold 0 \
        --input-drop 0 \
        --att-drop 0 \
        --label-drop 0 \
        --dropout 0.5 \
        --pre-process \
        --eval 1 \
        --act sigmoid \
        --batch 5000 \
        --patience 300 \
        --n-layers-2 6 \
        --label-num-hops 9 \
        --num-hops 6 \
        --hidden 1024 \
        --bns \
        --temp 0.001 \
        |& tee ./gamlp.${dataset}.giant-xrt.log
    ##
else
    echo "dataset=${dataset} is NOT valid! try ogbn-products or ogbn-papers100M"
    exit
fi

