import os
import pandas as pd
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer
from transformers import pipeline, BitsAndBytesConfig
from peft import PeftModel
import torch
import logging
from tqdm import tqdm
from configs import modeltype2path
import warnings
from itertools import product
from huggingface_hub import interpreter_login


token = '' # Add your HuggingFace token

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")

save_directory = "/gpfs/home/dcolombaro/Jailbreak_LLM-main/models/8bit quantized fine tuned"
model_dir = "/gpfs/home/dcolombaro/Jailbreak_LLM-main/models"


def main():

    quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
    )

    model = AutoModelForCausalLM.from_pretrained(
            model_dir, # save_directory
            quantization_config = quant_config, # comment when loading the original model
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            token = token)

    tokenizer = AutoTokenizer.from_pretrained(
            model_dir, # save_directory 
            low_cpu_mem_usage=True,
            device_map="auto",
            token = token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # model.push_to_hub("danielcolombaro/llama2-7b-int8-adversarial", token=token)

    check_dir = '/gpfs/home/dcolombaro/Jailbreak_LLM-main/fine tuning/results/checkpoint-350'
    merge_model = PeftModel.from_pretrained(model, check_dir)
    merge_model = merge_model.merge_and_unload()
    merge_model.save_pretrained(save_directory + '/prova')
    tokenizer.save_pretrained(save_directory + '/prova')

if __name__ == "__main__":
    main()

