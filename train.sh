#!/bin/bash
#SBATCH --job-name=bsc_transformer       # naam van je job in de queue
#SBATCH --output=logs/%x_%j.out          # stdout → logs/bsc_transformer_<JOBID>.out
#SBATCH --error=logs/%x_%j.err           # stderr → logs/bsc_transformer_<JOBID>.err
#SBATCH --time=02:00:00                  # wall‐time (HH:MM:SS)
#SBATCH -N 1                             # 1 node
#SBATCH --ntasks-per-node=1              # 1 task per node
#SBATCH -p defq                          # default GPU‐queue (TitanX, 15 min limiet doordeweeks)
#SBATCH --gres=gpu:1                     # vraag 1 GPU aan

#### 1) Load shell‐config en CUDA ####
. /etc/bashrc
. /etc/profile.d/lmod.sh
module load cuda/12.6/toolkit            # kies de versie die je hebt geïnstalleerd

#### 2) Activateer je Conda‐omgeving ####
source /var/scratch/$USER/anaconda3/etc/profile.d/conda.sh
conda activate pytorch-gpu               # env met Python 3.12, torch+cu126

#### 3) Ga naar je project ####
cd /var/scratch/$USER/bachelor_thesis

#### 4) Maak folders voor logs/checkpoints ####
mkdir -p logs
mkdir -p results/exp1

#### 5) Run je training ####
python main.py \
  --num-train 200 \
  --epochs 10 \
  --batch-size 128 \
  --seq-len 256 \
  --emb-size 512 \
  --n-layers 6 \
  --n-heads 8 \
  --ff-hidden-mult 4 \
  --dropout 0.1 \
  --lr 1e-4 \
  --output-dir results/exp1

