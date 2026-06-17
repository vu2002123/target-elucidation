#!/bin/bash
#SBATCH --job-name=gnina_array
#SBATCH --output=logs/dock_%j_%a.out
#SBATCH --error=logs/dock_%j_%a.err
#SBATCH --array=1-5
#SBATCH --cpus-per-task=48
#SBATCH --mem=20G
#SBATCH --time=72:00:00
#SBATCH --partition=defq
#SBATCH --gres=gpu:gpu:1

NUM_JOBS=5
module load anaconda
eval "$(conda shell.bash hook)"
conda activate chemtools_12.9
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

RESULT_DIR="job_${SLURM_ARRAY_JOB_ID}"
mkdir -p logs
mkdir -p $RESULT_DIR

if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    find . -name "*config.txt" > logs/all_configs_${SLURM_ARRAY_JOB_ID}.txt
    # Split the master list into N chunks (chunk_00, chunk_01, etc.)
    split -d --numeric-suffixes=1 -a 2 --number=l/$NUM_JOBS --additional-suffix=.txt logs/all_configs_${SLURM_ARRAY_JOB_ID}.txt logs/${SLURM_ARRAY_JOB_ID}_chunk_
fi

sleep 5

# Format the ID to match split's suffix (0 becomes 00, 1 becomes 01)
CHUNK_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
MY_CHUNK="logs/${SLURM_ARRAY_JOB_ID}_chunk_${CHUNK_ID}.txt"

# -j: Number of simultaneous runs (matches allocated CPUs)
cat "$MY_CHUNK" | parallel -j $SLURM_CPUS_PER_TASK \
    "gnina --config {}"
