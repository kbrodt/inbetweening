#!/usr/bin/env sh


set -eu  # o pipefail

GPU=${GPU:-0,1,2,3}
PORT=${PORT:-29500}
N_GPUS=${N_GPUS:-4}

OPTIM=adamw
LR=0.0001
#LR=0.00005
#LR=0.0005
WD=0.01

N_EPOCHS=200
T_MAX=200
loss=xent
attn=scse
img=512
dsmult=2
nclass=2

#backbone=tf_efficientnetv2_s_in21k
backbone=vit_large_patch14_dinov2
#backbone=tf_efficientnetv2_m_in21k
loss=bce
N_EPOCHS=400
T_MAX=400
FOLD=0
NFOLDS=5
data_dir=$SLURM_TMPDIR/data
test_data_dir=$SLURM_TMPDIR/test
train_data_dir=$SLURM_TMPDIR/train
chkps_dir=~/scratch/overlap
BS=4
CHECKPOINT="${chkps_dir}"/chkps/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_"${img}"_sreal10_segmpov_ft_normss

#MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
python \
    ./src/train.py \
        --img-dir "${data_dir}" \
        --train-img-dir "${train_data_dir}" \
        --test-img-dir "${test_data_dir}" \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 3 \
        --n-classes "${nclass}" \
        --ds-mult "${dsmult}" \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --T-max "${T_MAX}" \
        --num-epochs "${N_EPOCHS}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --n-folds "${NFOLDS}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT/model_last.pth \
        --img-size $img $img \
        --fp16 \
        --resume \


        # --dec-attn-type $attn \
        # --dec-channels 256 240 224 208 192 \
        # --load ./chkps/tf_efficientnetv2_s_in21ft1k_f0_b20_e200_attn/model_last.pth \
