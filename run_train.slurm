#!/bin/bash
#SBATCH --job-name=malaria_train
#SBATCH --output=malaria_train_%j.out
#SBATCH --error=malaria_train_%j.err
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-16:1
#SBATCH --time=12:00:00

# Load modules
module purge
module load anaconda3

# Activate conda environment
conda activate pytorch

# Move to the project directory
cd $SLURM_SUBMIT_DIR

# Run the training script
python train.py
