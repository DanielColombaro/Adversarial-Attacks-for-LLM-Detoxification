import pandas as pd
from datasets import Dataset
import json

with open("C:\\Users\\danie\\Documents\\tirocinio\\NLP Bcn\\Persona modulation\\Jailbreak_LLM-main\\data\\advbench.txt") as f:
    prompts = f.readlines()[:100]

responses = pd.read_csv("C:\\Users\\danie\\Documents\\tirocinio\\NLP Bcn\\Persona modulation\\Jailbreak_LLM-main\\merged_eval_results\\quantized fine tuned\\exploited\\Llama-2-7b-chat-hf_sample_5_advbench_matching_only_text.csv")
responses = responses['temp_0.8']# Define a function to transform the data

labels = pd.read_csv("C:\\Users\\danie\\Documents\\tirocinio\\NLP Bcn\\Persona modulation\\Jailbreak_LLM-main\\merged_eval_results\\quantized fine tuned\\exploited\\Llama-2-7b-chat-hf_sample_5_advbench_matching_only.csv")
labels = labels['temp_0.75']

data = pd.DataFrame({'prompts':prompts, 'outputs':responses, 'labels':labels})
# Specify the filename
filename = 'C:\\Users\\danie\\Documents\\tirocinio\\NLP Bcn\\Persona modulation\\Jailbreak_LLM-main\\data\\advbench_complete.csv'

data.to_csv(filename)