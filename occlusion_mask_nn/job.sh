#!/bin/bash
#SBATCH --gpus-per-node=a100:1  # Request GPU "generic resources"
#SBATCH --cpus-per-task=2       # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=32000M           # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-02:00          # DD-HH:MM:SS

#module load python/3.8
#cuda cudnn

SOURCEDIR=~/overlap
data_dir=$SLURM_TMPDIR/data

source ~/ENV/bin/activate
#virtualenv --no-download $SLURM_TMPDIR/env
#source $SLURM_TMPDIR/env/bin/activate
#pip install --no-index --upgrade pip setuptools
#pip install --no-index -r $SOURCEDIR/requirements.txt
#pip install --no-index --no-deps -v -e $SOURCEDIR/segmentation_models.pytorch

unzip -u -d $data_dir $SOURCEDIR/sketch_overlap_synth.zip
unzip -u -d $SLURM_TMPDIR/test $SOURCEDIR/sketches.zip
unzip -u -d $SLURM_TMPDIR/train $SOURCEDIR/sketch.zip

sh $SOURCEDIR/run.sh
