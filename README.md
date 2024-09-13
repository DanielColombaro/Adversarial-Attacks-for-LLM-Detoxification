# Catastrophic Jailbreak of Open-source LLMs via Exploiting Generation

DISCLAIMER: The experiments conducted contain harmful and dangerous content that shall be used only for research purposes.

This is the official repository for "Adversarial Attacks for LLM Jailbreaking", the research work I carried out for my Master’s thesis at Università di Milano Bicocca (Milan, IT). The experiments were carried out at the TALN Laboratory of Universitat Pompeu Fabra (Barcelona, ES) and made possible by the SNOW high performance cluster (HPC).

In this work, I stack the generation exploitation attack with quantization and fine-tuning processes to maximize the success of the adversarial attack. The target model is primarily llama2-7b-chat-hf, but the experiments were extended to the most recent Italian LLMs, as of May 2024.

If you find this implementation helpful, please consider citing this repository.

## Table of Contents
- [Preparation](https://github.com/DanielColombaro/Adversarial-Attacks-for-LLM-Detoxification#preparation)
- [Launch the attack](https://github.com/DanielColombaro/Adversarial-Attacks-for-LLM-Detoxification#launch-the-attack)
- [Evaluate the attack](https://github.com/DanielColombaro/Adversarial-Attacks-for-LLM-Detoxification#evaluate-the-attack)
- [Finetuning](https://github.com/DanielColombaro/Adversarial-Attacks-for-LLM-Detoxification#finetuning)

## Preparation

To launch the attack on a specified model, download the model and assign its path to the variable ‘save_directory’ inside ‘attack.py’.



## Launch the attack

You can either directly run ‘attack.py’ or, if you have access to a HPC for enhanced resources working with SLURM, you can submit a job through the ‘sbatch’ command and a Bash file. Inside the Bash file there are all the computational requirements specifications such as the number of nodes, the priority queue, the time limit, the number of GPUs and their memory storage, as well as the output and errors log files.
To successfully execute Python code within a SLURM job, you must first create a new Conda environment. This is achieved by running an interactive job, according to the following instructions:
* From the terminal of a (virtual) machine connected to SLURM, run ‘salloc’.
* Run ‘eval "$(conda shell.bash hook)"’.
* Run ‘conda create -n environment’ and change ‘environment’ with a custom name.
* Run ‘pip install --no-index -r requirements.txt’ to install all required modules from a local file in the same folder.
* Run ‘pip install module’ to install a specific module; replace ‘module’ with a selected one.
* Exit from the interactive mode.
* Run ‘nano submit.sh’.
* In ‘conda activate environment’ substitute ‘environment’ with the newly created environment name.
* Exit from the editing mode of the Bash file with Ctrl+C.
* Run ‘submit.sh’.


### Default or greedy generation

Inside ‘attack.py’ there is a dictionary of keyword arguments ‘kargs’. By setting ‘use_default=True’ you run the default decoding configuration (temperature=0.1, top_p=0.9. To generate text samples with greedy decoding, simply set ‘use_greedy=True’ inside the same dictionary.

### Exploited generation

The exploited decoding generation is activated by setting the flags ``` 
    --tune_temp=True \
    --tune_topp=True \
    --tune_topk=True \
``` in the ‘kargs’ dictionary.

You can increase the `--n_sample` parameter to generate more examples for each prompt, which potentially makes the attack stronger.

## Evaluate the attack

To evaluate the attack with substring matching, you can:
* Edit the Bash file with ‘nano submit_eval.sh’.
* Add the flag ‘-matching_only’ in the line starting with ‘python evaluation.py’. Save and exit the edit mode.
* To evaluate text generated with default and greedy decoding respectively, set ‘config default-only’ or ‘config greedy only’ respectively in the same line.
* The full list of configurable arguments is specified in the ‘args’ dictionary inside the file ‘evaluation.py’.
* Execute a job through ‘sbatch submit_eval.sh’. The complete procedure to run a SLURM job is detailed in the section ‘Launch the Attack’.
* To evaluate text in Italian, in ‘evaluation.py’ at line 170 set ‘ita=True’ inside ‘not_matched()’. Furthermore, at line 250 substitute the path “./data/advbench.txt" with “./data/advbench_ita.txt”. At line 253, perform the same operation for the MaliciousInstruct dataset.

The results are summarized in a .json file:
```python
{
    "best_attack_config": {
        "temp": {
            "temp_0.95": "25"   # Most vulnerable temp is 0.95, which gives 25% ASR
        },
        "topk": {
            "topk_500": "26"   # Most vulnerable top-k is 500, which gives 26% ASR
        },
        "topp": {
            "topp_0.7": "29"   # Most vulnerable temp is 0.7, which gives 29% ASR
        }
    },
    "greedy": "16",            # Greedy decoding without system prompt gives 16% ASR
    "break_by_temp": "47",     # Exploiting temp only gives 47% ASR
    "break_by_topk": "54",     # Exploiting top-k only gives 54% ASR
    "break_by_topp": "77",     # Exploiting top-p only gives 77% ASR
    "break_by_all": "81"       # Exploiting all decoding strategies gives 81% ASR
}
``` 

You can aggregate results for different models (to obtain Table 1 in our paper) using the example shown in `aggregate_results.ipynb`.

## Finetuning

To enhance the attack with finetuning, do the following:
* Run ‘cd “fine tuning”’.
* Run ‘nano SFT_Trainer.py’.
* Modify the following variables:
o ‘data_dir’: directory containing the training data
o ‘model_dir’: directory or HuggingFace repository containing the original model
o ‘save_dir’: directory to save the modified model tensors
o ‘dataset’: define the specific training dataset
* Exit edit mode.
* Run ‘sbatch submit_tuning.sh’.

