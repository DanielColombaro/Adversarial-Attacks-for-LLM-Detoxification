#!/bin/bash
#SBATCH -J eval_advbench
#SBATCH -N 1
#SBATCH -p medium
#SBATCH --chdir /gpfs/home/dcolombaro/Jailbreak_LLM-main
#SBATCH -o outfile_eval_ita
#SBATCH -e errfile_eval_ita
#SBATCH --gres=gpu:1
# SBATCH --mem=30g
#SBATCH --time 4:00:00

module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate environment # dec_jailbreak

python evaluation.py --config exploited --matching_only --model gpfs/home/dcolombaro/Jailbreak_LLM-main/models/Llamantino3-8b/merged --n_sample 5 --n_eval 500 --use_advbench

pip install --no-index --upgrade pip
pip install --no-index -r requirements.txt
python -m pip install --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org --trusted-host pypi.python.org/pypi/openpyxl --upgrade pip

