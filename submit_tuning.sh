#!/bin/bash
#SBATCH -J Llama2_7b_fine_tuning
#SBATCH -N 1
#SBATCH -p medium
#SBATCH --chdir "/gpfs/home/dcolombaro/Jailbreak_LLM-main"
#SBATCH -o outfile_ft
#SBATCH -e errfile_ft
#SBATCH --gres=gpu:2
#SBATCH --mem=24g
# SBATCH --time 14-00:00:00

module load Miniconda3/4.9.2
#module load CUDA/12.1.0
eval "$(conda shell.bash hook)"
conda activate environment # RL


# export TORCH_DISTRIBUTED_DEBUG=INFO
# export NCCL_DEBUG=INFO

#pip install datasets numpy pandas torch transformers accelerate trl bitsandbytes openpyxl psutil
pip install --no-index --upgrade pip
pip install --no-index -r requirements_internet.txt # togli internet
python -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --upgrade pip

python SFT_Trainer.py 

