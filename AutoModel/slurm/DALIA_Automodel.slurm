#!/bin/bash
#SBATCH --job-name=bc
#SBATCH --output=logs/out/DALIA_AM_%j.out 
#SBATCH --error=logs/err/DALIA_AM_%j.err
#SBATCH --gpus-per-node=4
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1 
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=36 

## load module 
module purge
module load slurm/slurm/24.11

# --------- ENV DISTRIB ----------
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NUM_GPUS=4                      # GPUs par nœud
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))

#export CUDA_DEVICE_MAX_CONNECTIONS=1
#export TORCH_NCCL_AVOID_RECORD_STREAMS=1
#export NCCL_NVLS_ENABLE=1              # ou 0 si tu veux désactiver NVLS pour debug

# HF / W&B etc
#export HF_HOME=/lustre/$USER/hf_home
#export HF_TOKEN=xxx
#export WANDB_API_KEY=yyy
export HF_DATASETS_CACHE="/lustre/work/sos/ssos040/cache_hf"

# ton image Singularity
IMAGE=$WORK/nemo2506_arm.sif
export APPTAINER_BINDPATH="$WORK,$ALL_WORK"

# configuration Automodel (sans section slurm, ou ignorée)
CFG=/lustre/$USER/configs/llm_finetune.yaml

# commande à exécuter dans le conteneur
read -r -d '' CMD <<EOF
cd /workspace/Automodel               # ou répertoire où le code est monté
echo "HOST=\$(hostname)  RANK=\$SLURM_PROCID  LOCAL_RANK=\$SLURM_LOCALID"
torchrun \\
  --nnodes=\$SLURM_NNODES \\
  --nproc-per-node=\$NUM_GPUS \\
  --rdzv_backend=c10d \\
  --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT \\
  \$MYWORK/AutoModel/patched_train_ft.py -c $CFG
EOF

echo "$CMD"

srun \
  apptainer exec --nv \
    "$IMAGE" \
    bash -c "$CMD"
