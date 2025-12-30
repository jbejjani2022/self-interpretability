#!/bin/bash
#SBATCH --job-name=api_introspect
#SBATCH --account=sham_lab
#SBATCH --partition=sapphire
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-12:00:00
#SBATCH --mem=128G
#SBATCH -o api_output/job.%N.%j.out          # STDOUT
#SBATCH -e api_error/job.%N.%j.err           # STDERR
#SBATCH --mail-user=jbejjani@college.harvard.edu
#SBATCH --mail-type=ALL

# Load modules
module load python/3.10.13-fasrc01
# module load cuda/12.9.1-fasrc01
# module load cudnn/9.10.2.21_cuda12-fasrc01

# Activate conda environment
mamba deactivate
mamba activate /n/holylabs/LABS/sham_lab/Users/jbejjani/envs/introspection

cd ..

python api.py
