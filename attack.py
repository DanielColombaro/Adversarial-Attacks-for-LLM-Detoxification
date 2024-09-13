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

token = '' # Add your HuggingFace token

logging.basicConfig(level=logging.INFO)
warnings.simplefilter("ignore")

kargs = {'model': "Llama-2-7b-chat-hf",
        'n_sample': 5,
        'use_greedy': False,
        'use_default': False,
        'tune_temp': False,
        'tune_topp': False,
        'tune_topk': False,
	    'tune_joint': True,
        'use_advbench': True,
        'use_quantized': False,
        'use_tuned': True}

def get_sentence_embedding(model, tokenizer, sentence):
    sentence = sentence.strip().replace('"', "")
    word_embeddings = model.get_input_embeddings()

    # Embed the sentence
    tokenized = tokenizer(sentence, return_tensors="pt", add_special_tokens=False).to(
        model.device
    )
    embedded = word_embeddings(tokenized.input_ids)
    return embedded


def main():

    model_name = kargs['model']

    WEIGHTS_PATH = model_name
    TOKENIZER_PATH = WEIGHTS_PATH

    save_directory = "/gpfs/home/dcolombaro/Jailbreak_LLM-main/models"
    fname = kargs['model']

    if kargs['n_sample'] > 1:
        fname += f"_sample_{kargs['n_sample']}"
    if kargs['use_advbench']:
        fname += "_advbench"
    if not os.path.exists(f"outputs/{fname}"):
        os.makedirs(f"outputs/{fname}")
    if kargs['use_quantized']:
        fname = "quantized/" + fname
        save_directory += "/8bit quantized"
    if kargs['use_tuned']:
        fname = "quantized fine tuned/" + fname
        save_directory += "/8bit quantized fine tuned"

## Quantization configuration
    quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)


    model = AutoModelForCausalLM.from_pretrained(
    	    save_directory,
            # quantization_config = quant_config, # comment when loading the original model
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            token = token)

    tokenizer = AutoTokenizer.from_pretrained(
            save_directory, 
            low_cpu_mem_usage=True,
            device_map="auto",
            token = token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    logging.info(f"Model size: {model.get_memory_footprint()/1e9}")
    logging.info(f"Model name: {fname}")

    if kargs['use_advbench']:
        with open("./data/advbench.txt") as f:
            lines = f.readlines()[:100] # onyl consider the first 100 instances
    else:
        with open("./data/MaliciousInstruct.txt") as f:
            lines = f.readlines()

    #lines = [prepend_sys_prompt(l, kargs) for l in lines]

    if kargs['use_greedy']:
        logging.info(f"Running greedy")
        prompts = []
        outputs = []
        model.eval()

        for sentence in tqdm(lines):
            try:
                ground_truth_embeds = get_sentence_embedding(
                    model, tokenizer, sentence
                    )
                ground_truth_generation = model.generate(
                    inputs_embeds=ground_truth_embeds,
                    max_new_tokens=128,
                    do_sample=False,
                    num_return_sequences=1)

                ground_truth_generation = tokenizer.batch_decode(
                    ground_truth_generation)

                outputs.extend(ground_truth_generation)
                prompts.extend([sentence] * kargs['n_sample'])
            except:
                continue
            results = pd.DataFrame()
            results["prompt"] = [line.strip() for line in prompts]
            results["output"] = outputs
            results.to_csv(f"outputs/{fname}/output_greedy.csv")

    if kargs['use_default']:
        logging.info(f"Running default, top_p=0.9, temp=0.1")
        prompts = []
        outputs = []
        model.eval()

        for sentence in tqdm(lines):
            try:
                ground_truth_embeds = get_sentence_embedding(
                    model, tokenizer, sentence)
                ground_truth_generation = model.generate(
                    inputs_embeds=ground_truth_embeds,
                    max_new_tokens=128,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    top_k=200,
                    num_return_sequences=kargs['n_sample'])

                ground_truth_generation = tokenizer.batch_decode(
                    ground_truth_generation)

                outputs.extend(ground_truth_generation)
                prompts.extend([sentence] * kargs['n_sample'])

            except:
                continue
            results = pd.DataFrame()
            results["prompt"] = [line.strip() for line in prompts]
            results["output"] = outputs
            results.to_csv(f"outputs/{fname}/output_default.csv")

    if kargs['tune_temp']:
        for temp in np.arange(0.7, 1.05, 0.05): 
            temp = np.round(temp, 2)
            logging.info(f"Running temp = {temp}")
            prompts = []
            outputs = []
            model.eval()

            for sentence in tqdm(lines):
                try:
                    ground_truth_embeds = get_sentence_embedding(
                        model, tokenizer, sentence)
                    ground_truth_generation = model.generate(
                        inputs_embeds=ground_truth_embeds,
                        max_new_tokens=128,
                        temperature=temp,
                        do_sample=True,
                        num_return_sequences=kargs['n_sample'])

                    ground_truth_generation = tokenizer.batch_decode(
                        ground_truth_generation)

                    outputs.extend(ground_truth_generation)
                    prompts.extend([sentence] * kargs['n_sample'])

                except:
                    continue
                results = pd.DataFrame()
                results["prompt"] = [line.strip() for line in prompts]
                results["output"] = outputs
                results.to_csv(f"outputs/{fname}/output_temp_{temp}.csv")

    if kargs['tune_topp']:
        for top_p in np.arange(0.7, 1.05, 0.05): # cambiare con 0.7 starting point
            top_p = np.round(top_p, 2)
            logging.info(f"Running topp = {top_p}")
            outputs = []
            prompts = []
            model.eval()

            for sentence in tqdm(lines):
                try:
                    ground_truth_embeds = get_sentence_embedding(
                        model, tokenizer, sentence)

                    ground_truth_generation = model.generate(
                        inputs_embeds=ground_truth_embeds,
                        max_new_tokens=128,
                        top_p=top_p,
                        do_sample=True,
                        num_return_sequences=kargs['n_sample'])

                    ground_truth_generation = tokenizer.batch_decode(
                        ground_truth_generation)

                    outputs.extend(ground_truth_generation)
                    prompts.extend([sentence] * kargs['n_sample'])

                except:
                    continue
                results = pd.DataFrame()
                results["prompt"] = [line.strip() for line in prompts]
                results["output"] = outputs
                results.to_csv(f"outputs/{fname}/output_topp_{top_p}.csv")

    if kargs['tune_topk']:
        for top_k in [1, 2, 5, 10, 20, 50, 100, 200, 500]:
            logging.info(f"Running topk = {top_k}")
            outputs = []
            prompts = []
            model.eval()

            for sentence in tqdm(lines):
                try:
                    ground_truth_embeds = get_sentence_embedding(
                        model, tokenizer, sentence)

                    ground_truth_generation = model.generate(
                        inputs_embeds=ground_truth_embeds,
                        max_new_tokens=128,
                        top_k=top_k,
                        do_sample=True,
                        num_return_sequences=kargs['n_sample'])

                    ground_truth_generation = tokenizer.batch_decode(
                        ground_truth_generation)

                    outputs.extend(ground_truth_generation)
                    prompts.extend([sentence] * kargs['n_sample'])

                except:
                    continue
                results = pd.DataFrame()
                results["prompt"] = [line.strip() for line in prompts]
                results["output"] = outputs
                results.to_csv(f"outputs/{fname}/output_topk_{top_k}.csv")

    if kargs['tune_joint']:
        values_temp = [0.75] #[0.9, 0.95, 1.0]
        values_topp = [1.0] #[0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        values_topk = [20] #[5,10,20,50,100,200,500]
        combinations = product(values_temp, values_topp, values_topk)
        print(combinations)
        for combo in combinations:
            rounded_combo = tuple(np.round(i, 2) for i in combo)
            logging.info(f"Running configuration: {rounded_combo}")
            prompts = []
            outputs = []
            model.eval()
            for sentence in tqdm(lines):
                temperature=float(rounded_combo[0])
                top_p=float(rounded_combo[1])
                top_k=int(rounded_combo[2])
                print(sentence)
                try:
                    ground_truth_embeds = get_sentence_embedding(
                        model, tokenizer, sentence)
                    ground_truth_generation = model.generate(
                        inputs_embeds=ground_truth_embeds,
                        max_new_tokens=128,
                        temperature = temperature,
                        top_p = top_p,
                        top_k = top_k,
                        do_sample=True,
                        num_return_sequences=kargs['n_sample'])
                    ground_truth_generation = tokenizer.batch_decode(ground_truth_generation)

                    outputs.extend(ground_truth_generation)
                    prompts.extend([sentence] * kargs['n_sample'])

                except:
                    continue
                results = pd.DataFrame()
                results["prompt"] = [line.strip() for line in prompts]
                results["output"] = outputs
                results.to_csv(f"outputs/{fname}/joint/output_joint_temp{temperature}_topp{top_p}_topk{top_k}.csv")

if __name__ == "__main__":
    main()
