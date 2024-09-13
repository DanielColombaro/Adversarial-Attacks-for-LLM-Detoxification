#!/bin/bash
#SBATCH -J attack_best
#SBATCH -N 1
#SBATCH -p high
#SBATCH --chdir "/gpfs/home/dcolombaro/Jailbreak_LLM-main"
#SBATCH -o outfile_new_tuned
#SBATCH -e errfile_new_tuned
#SBATCH --gres=gpu:1
#SBATCH --mem=30g
#SBATCH --time 14-00:00:00

module load Miniconda3/4.9.2
#module load CUDA/12.1.0
eval "$(conda shell.bash hook)"

conda activate environment # RL

python attack.py


# %%capture

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --upgrade pip

