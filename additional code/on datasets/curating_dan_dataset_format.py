from datasets import load_dataset
import re
import pandas as pd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the dataset
dataset = pd.read_excel("/gpfs/home/dcolombaro/Jailbreak_LLM-main/data/DAN dataset completed.xlsx")

token = ''
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M", src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

def clean_string(text):
    text = text.replace('\n', '').split()
    text = ' '.join(text)
    return text

# Define a function to transform the data
def transform_conversation(example):
    input_prompt = tokenizer(example['prompt'], return_tensors='pt')
    completion_prompt = model.generate(**input_prompt, forced_bos_token_id=tokenizer.lang_code_to_id["ita_Latn"])
    output_prompt = tokenizer.batch_decode(completion_prompt, skip_special_tokens=True)[0]

    input_response = tokenizer(example['output'], return_tensors='pt')
    completion_response = model.generate(**input_response, forced_bos_token_id=tokenizer.lang_code_to_id["ita_Latn"])
    output_response = tokenizer.batch_decode(completion_response, skip_special_tokens=True)[0]

    output_prompt = output_prompt.strip()
    output_response = output_response.strip()
    
    print(output_prompt, output_response)

    # Apply the new template
    reformatted_text = f'[INST] {clean_string(output_prompt)} [/INST] {clean_string(output_response)}'

    return {'text': reformatted_text}

# Apply the transformation
transformed_dataset = dataset.apply(transform_conversation, axis=1).tolist()
print(transformed_dataset[0])

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        text = f"### Domanda: {clean_string(example['prompt'][i])} ### Risposta: {clean_string(example['output'][i])}"
        output_texts.append(text)
    return output_texts

transformed_dataset = formatting_prompts_func(transformed_dataset)

# Initialize an empty list to store the text strings
text_list = []

# Iterate over the inner dictionaries
for value in transformed_dataset:
    # Append the text value to the list
    text_list.append(value['text'])
    print(text_list)

# Join all the text strings into a single string
text_string = ' '.join(text_list)
# Create a new dictionary with the concatenated text string
new_dict = {'text': text_string}

new_dict.keys()
save_dir = '/gpfs/home/dcolombaro/Jailbreak_LLM-main/data/'
with open(save_dir + "data_new_ita.jsonl", "w") as outfile:
    json.dump(transformed_dataset, outfile)

